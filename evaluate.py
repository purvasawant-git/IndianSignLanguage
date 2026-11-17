"""
Root entry point to run Performance Evaluation for Section 4.6
Usage:
    python evaluate.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from evaluate_performance import main

if __name__ == '__main__':
    main()
