import os
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.text import PP_ALIGN

def create_ppt():
    dataset_path = 'dataset_ISL'
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found.")
        return

    prs = Presentation()
    
    # Title Slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "ISL Alphabet Dataset Representations"
    subtitle.text = "Expected hand gestures for A-Z"
    
    classes = sorted([c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))])
    
    print("Generating slides...")
    for c in classes:
        class_path = os.path.join(dataset_path, c)
        files = os.listdir(class_path)
        if not files:
            continue
            
        # Get the first image
        img_path = os.path.join(class_path, files[0])
        
        # Add a slide (layout 5 is title only)
        slide_layout = prs.slide_layouts[5]
        slide = prs.slides.add_slide(slide_layout)
        
        # Set title
        title_shape = slide.shapes.title
        title_shape.text = f"Letter: {c.upper()}"
        title_shape.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        
        # Add image
        # Center the image roughly
        left = Inches(3)
        top = Inches(2)
        height = Inches(4)
        try:
            slide.shapes.add_picture(img_path, left, top, height=height)
        except Exception as e:
            print(f"Failed to add image for class {c}: {e}")
        
    output_path = "ISL_Alphabet_Representations.pptx"
    prs.save(output_path)
    print(f"Presentation saved to {output_path}")

if __name__ == '__main__':
    create_ppt()
