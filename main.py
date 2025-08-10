import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.enhanced_streamlit_app import main

if __name__ == "__main__":
    main() 