import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

class LandmarkPoint:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

class LandmarkExtractor:
    def __init__(self, static_image_mode=True, max_num_hands=2, min_detection_confidence=0.35, enable_smoothing=True):
        # Path to hand_landmarker.task
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hand_landmarker.task')
        if not os.path.exists(model_path):
            model_path = 'hand_landmarker.task'
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        running_mode = vision.RunningMode.IMAGE
        
        # High-sensitivity detection confidence for reliable 2-hand capture
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.enable_smoothing = enable_smoothing
        self.prev_hands = []        # Cache of previous smoothed hand objects
        self.smoothing_alpha = 0.70 # Responsiveness vs stability weight
        self.two_hand_grace = 0     # Grace frame counter to prevent 2-hand flicker

    def normalize_hand(self, hand):
        """
        Normalizes a single hand's 21 landmarks:
        - Translates wrist (landmark 0) to (0, 0, 0).
        - Scales coordinates by max euclidean distance from wrist to any landmark.
        """
        wrist = (hand[0].x, hand[0].y, hand[0].z)
        max_dist = 0.0
        for lm in hand:
            d = np.sqrt((lm.x - wrist[0])**2 + (lm.y - wrist[1])**2 + (lm.z - wrist[2])**2)
            if d > max_dist:
                max_dist = d
        if max_dist < 1e-6:
            max_dist = 1.0

        norm_lms = []
        for lm in hand:
            nx = (lm.x - wrist[0]) / max_dist
            ny = (lm.y - wrist[1]) / max_dist
            nz = (lm.z - wrist[2]) / max_dist
            norm_lms.extend([nx, ny, nz])

        return norm_lms, wrist, max_dist

    def compute_geometric_features(self, hand_landmarks, wrist, scale):
        """
        Computes invariant special geometric coordinates for a single hand:
        - 5 Fingertip to wrist distances
        - 5 Finger extension ratios (tip-to-wrist / mcp-to-wrist)
        - 10 Pairwise fingertip distances (4-8, 4-12, ..., 16-20)
        - 5 Joint bending angles (cosine at PIP knuckle)
        Total = 25 distinctive invariant features.
        """
        pts = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks])
        w = np.array(wrist)
        
        tip_indices = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
        mcp_indices = [2, 5, 9, 13, 17]  # Base knuckles
        pip_indices = [1, 6, 10, 14, 18] # Intermediate joints
        
        # 1. Fingertip to wrist distances (scaled)
        tip_dists = [float(np.linalg.norm(pts[t] - w) / scale) for t in tip_indices]
        
        # 2. Finger extension ratios (tip-wrist dist / mcp-wrist dist)
        ext_ratios = []
        for t, m in zip(tip_indices, mcp_indices):
            d_mcp = np.linalg.norm(pts[m] - w)
            d_tip = np.linalg.norm(pts[t] - w)
            ext_ratios.append(float(d_tip / (d_mcp + 1e-5)))
            
        # 3. Pairwise fingertip distances (10 combinations)
        pair_dists = []
        for i in range(len(tip_indices)):
            for j in range(i + 1, len(tip_indices)):
                d_ij = np.linalg.norm(pts[tip_indices[i]] - pts[tip_indices[j]]) / scale
                pair_dists.append(float(d_ij))
                
        # 4. Joint bending angles (cosine of angle at PIP joint: MCP -> PIP -> Tip)
        bend_angles = []
        for m, p, t in zip(mcp_indices, pip_indices, tip_indices):
            v1 = pts[m] - pts[p]
            v2 = pts[t] - pts[p]
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
            cos_theta = np.dot(v1, v2) / denom
            bend_angles.append(float(np.clip(cos_theta, -1.0, 1.0)))
            
        return tip_dists + ext_ratios + pair_dists + bend_angles

    def extract_landmarks(self, image, try_padding=True):
        """
        Extracts hand landmarks with spatial left-right sorting and proximity-based tracking.
        Smooths each hand independently to prevent glitching when a second hand appears.
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = self.detector.detect(mp_image)

        # Fallback for tightly cropped dataset images
        if (not results or not results.hand_landmarks) and try_padding:
            h, w = image.shape[:2]
            pad_h, pad_w = int(h * 0.4), int(w * 0.4)
            padded = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
            img_rgb_pad = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            mp_image_pad = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb_pad)
            results = self.detector.detect(mp_image_pad)

        if not results or not results.hand_landmarks:
            self.prev_hands = []
            self.two_hand_grace = 0
            return None, []

        raw_hands = results.hand_landmarks

        # Filter out degenerate/collapsed hand detections
        valid_hands = []
        for hand in raw_hands:
            wrist = hand[0]
            max_d = max(np.sqrt((lm.x - wrist.x)**2 + (lm.y - wrist.y)**2 + (lm.z - wrist.z)**2) for lm in hand)
            if max_d > 0.04:
                valid_hands.append(hand)

        if not valid_hands:
            self.prev_hands = []
            self.two_hand_grace = 0
            return None, []

        # Convert valid hands to LandmarkPoint objects and sort left-to-right by wrist X
        hand_objs = []
        for hand in valid_hands[:2]:
            hand_objs.append([LandmarkPoint(lm.x, lm.y, lm.z) for lm in hand])
            
        # CRITICAL FIX: Sort hands horizontally immediately so Hand 0 is always Left, Hand 1 is always Right
        hand_objs.sort(key=lambda h: h[0].x)

        # Anti-flicker grace logic: if 2 hands were active and 1 temporarily dips, preserve 2nd hand briefly
        if len(hand_objs) == 1 and len(self.prev_hands) == 2 and self.two_hand_grace < 2:
            self.two_hand_grace += 1
            # Check if remaining hand matches prev hand 0 or prev hand 1
            curr_wrist = hand_objs[0][0]
            d0 = np.sqrt((curr_wrist.x - self.prev_hands[0][0].x)**2 + (curr_wrist.y - self.prev_hands[0][0].y)**2)
            d1 = np.sqrt((curr_wrist.x - self.prev_hands[1][0].x)**2 + (curr_wrist.y - self.prev_hands[1][0].y)**2)
            if d0 < d1:
                # Current hand is Left hand, keep previous Right hand
                hand_objs = [hand_objs[0], self.prev_hands[1]]
            else:
                # Current hand is Right hand, keep previous Left hand
                hand_objs = [self.prev_hands[0], hand_objs[0]]
        elif len(hand_objs) == 2:
            self.two_hand_grace = 0
        else:
            self.two_hand_grace = 0

        # Apply Proximity-Based EMA Smoothing (matches each hand to its closest previous hand)
        if self.enable_smoothing and self.prev_hands:
            smoothed = []
            for cur_h in hand_objs:
                cur_wrist = cur_h[0]
                # Find nearest previous hand
                best_prev = None
                min_w_dist = 999.0
                for prev_h in self.prev_hands:
                    dist = np.sqrt((cur_wrist.x - prev_h[0].x)**2 + (cur_wrist.y - prev_h[0].y)**2)
                    if dist < min_w_dist:
                        min_w_dist = dist
                        best_prev = prev_h
                
                # Smooth if within continuity threshold
                if best_prev is not None and min_w_dist < 0.20:
                    a = self.smoothing_alpha
                    s_h = []
                    for lm_c, lm_p in zip(cur_h, best_prev):
                        sx = a * lm_c.x + (1 - a) * lm_p.x
                        sy = a * lm_c.y + (1 - a) * lm_p.y
                        sz = a * lm_c.z + (1 - a) * lm_p.z
                        s_h.append(LandmarkPoint(sx, sy, sz))
                    smoothed.append(s_h)
                else:
                    smoothed.append(cur_h)
            hand_objs = smoothed

        self.prev_hands = hand_objs

        # 1 Hand detected
        if len(hand_objs) == 1:
            norm_lms, wrist, scale = self.normalize_hand(hand_objs[0])
            geom_feats = self.compute_geometric_features(hand_objs[0], wrist, scale)
            hand1_features = norm_lms + geom_feats       # 88 features
            hand2_features = [0.0] * 88                  # 88 zeros
            cross_features = [0.0] * 6                   # 6 zeros
            landmarks = hand1_features + hand2_features + cross_features # 182 features
            return landmarks, hand_objs

        # 2 Hands detected (guaranteed sorted left to right)
        hand1, hand2 = hand_objs[0], hand_objs[1]

        norm_lms1, wrist1, scale1 = self.normalize_hand(hand1)
        norm_lms2, wrist2, scale2 = self.normalize_hand(hand2)

        # Relative offset of Hand 2 wrist from Hand 1 wrist
        norm_lms2[0] = (wrist2[0] - wrist1[0]) / scale1
        norm_lms2[1] = (wrist2[1] - wrist1[1]) / scale1
        norm_lms2[2] = (wrist2[2] - wrist1[2]) / scale1

        geom1 = self.compute_geometric_features(hand1, wrist1, scale1)
        geom2 = self.compute_geometric_features(hand2, wrist2, scale2)

        hand1_features = norm_lms1 + geom1  # 88 features
        hand2_features = norm_lms2 + geom2  # 88 features

        # Cross-hand interactions
        pts1 = np.array([(lm.x, lm.y, lm.z) for lm in hand1])
        pts2 = np.array([(lm.x, lm.y, lm.z) for lm in hand2])
        min_tip_d = min(np.linalg.norm(pts1[t1] - pts2[t2]) for t1 in [4,8,12,16,20] for t2 in [4,8,12,16,20])
        
        cross_features = [
            float(np.linalg.norm(pts1[0] - pts2[0]) / scale1),  # Wrist-to-wrist
            float(np.linalg.norm(pts1[8] - pts2[4]) / scale1),  # Left index to right thumb
            float(np.linalg.norm(pts1[4] - pts2[8]) / scale1),  # Left thumb to right index
            float(np.linalg.norm(pts1[8] - pts2[8]) / scale1),  # Left index to right index
            float(np.linalg.norm(pts1[12] - pts2[4]) / scale1), # Left middle to right thumb
            float(min_tip_d / scale1)                           # Closest fingertip touch
        ]

        landmarks = hand1_features + hand2_features + cross_features # 182 features
        return landmarks, hand_objs

    def draw_landmarks(self, image, hand_landmarks, hand_label=None):
        """
        Draws smoothed hand skeleton landmarks and connection bones on the image in-place using OpenCV.
        Features distinct vibrant finger color coding, glowing fingertip beacons, anti-aliased lines,
        and high-contrast dark underlay for clarity in any lighting environment.
        """
        h, w, _ = image.shape
        
        # 21 Landmark coordinate mapping
        points = []
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append((cx, cy))
            
        if len(points) < 21:
            return

        # Distinct color palette per finger branch (BGR format)
        FINGER_COLORS = {
            'thumb': (0, 215, 255),     # Vibrant Gold / Amber
            'index': (255, 240, 0),     # Electric Cyan
            'middle': (50, 255, 100),   # Neon Spring Green
            'ring': (0, 140, 255),      # Vivid Coral Orange
            'pinky': (235, 80, 255),    # Hot Pink / Magenta
            'palm': (220, 220, 220)     # Ice White / Palm Base
        }

        # Bone connection segments with their respective color category
        CONNECTION_GROUPS = [
            # Palm base
            ([(0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17), (1, 5), (2, 5)], FINGER_COLORS['palm']),
            # Thumb
            ([(1, 2), (2, 3), (3, 4)], FINGER_COLORS['thumb']),
            # Index
            ([(5, 6), (6, 7), (7, 8)], FINGER_COLORS['index']),
            # Middle
            ([(9, 10), (10, 11), (11, 12)], FINGER_COLORS['middle']),
            # Ring
            ([(13, 14), (14, 15), (15, 16)], FINGER_COLORS['ring']),
            # Pinky
            ([(17, 18), (18, 19), (19, 20)], FINGER_COLORS['pinky'])
        ]

        # Pass 1: Dark high-contrast shadow lines behind bones for visibility on any background
        for connections, _ in CONNECTION_GROUPS:
            for s, e in connections:
                if s < len(points) and e < len(points):
                    cv2.line(image, points[s], points[e], (20, 20, 20), 4, cv2.LINE_AA)

        # Pass 2: Vibrant anti-aliased bone segments
        for connections, color in CONNECTION_GROUPS:
            for s, e in connections:
                if s < len(points) and e < len(points):
                    cv2.line(image, points[s], points[e], color, 2, cv2.LINE_AA)

        # Fingertip indices: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
        FINGERTIP_MAP = {
            4: FINGER_COLORS['thumb'],
            8: FINGER_COLORS['index'],
            12: FINGER_COLORS['middle'],
            16: FINGER_COLORS['ring'],
            20: FINGER_COLORS['pinky']
        }

        # Pass 3: Landmark points (joints, wrist, and glowing fingertip beacons)
        for idx in range(len(points)):
            pt = points[idx]
            if idx in FINGERTIP_MAP:
                # Glowing radar beacon for fingertips
                tip_color = FINGERTIP_MAP[idx]
                cv2.circle(image, pt, 8, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(image, pt, 7, tip_color, 2, cv2.LINE_AA)
                cv2.circle(image, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(image, pt, 2, tip_color, -1, cv2.LINE_AA)
            elif idx == 0:
                # Wrist anchor landmark
                cv2.circle(image, pt, 7, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(image, pt, 5, (255, 220, 0), -1, cv2.LINE_AA)
                cv2.circle(image, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # Intermediate knuckles & joints
                cv2.circle(image, pt, 5, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(image, pt, 3, (0, 255, 220), -1, cv2.LINE_AA)
                cv2.circle(image, pt, 1, (255, 255, 255), -1, cv2.LINE_AA)

        # Optional stylish badge near wrist for multi-hand tracking
        if hand_label and len(points) > 0:
            wrist_x, wrist_y = points[0]
            tag_text = hand_label
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.45
            thick = 1
            (tw, th), baseline = cv2.getTextSize(tag_text, font, scale, thick)
            tag_x = max(5, wrist_x - tw // 2)
            tag_y = min(h - 10, wrist_y + 24)
            
            cv2.rectangle(image, (tag_x - 4, tag_y - th - 4), (tag_x + tw + 4, tag_y + baseline + 2), (20, 20, 20), -1)
            cv2.rectangle(image, (tag_x - 4, tag_y - th - 4), (tag_x + tw + 4, tag_y + baseline + 2), (0, 220, 255), 1)
            cv2.putText(image, tag_text, (tag_x, tag_y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
