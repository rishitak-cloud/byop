import streamlit as st
import pandas as pd
import numpy as np
from composition_rules import analyse
from nima_import import score
from PIL import Image

st.title("FrameAgent")
img = st.file_uploader(label="Choose a file", type=["jpg", "jpeg", "png", "webp"])
model = "/Users/rishitakandpal/Downloads/unet_model.pth"

if img is not None:
    image = Image.open(img)
    image.save("/Users/rishitakandpal/Downloads/photo.jpg")
    path = "/Users/rishitakandpal/Downloads/photo.jpg"
    st.image(img, "Uploaded file", width=300)
    #thirds, golden, centre, symm, line = analyse(path, model)
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    thirds, v1 = analyse(path, model).thirds()
    golden, v2 = analyse(path, model).golden()
    centre, v3 = analyse(path, model).centre()
    symm, v4 = analyse(path, model).symmetry()
    line, v5 = analyse(path, model).lines()
    sc = score(path)
    with col1: st.image(thirds, "Grid of Thirds", width=300) 
    with col2: st.image(golden, "Phi Grid for Golden Ratio", width=300)
    # with col3: st.image(centre, "Centroid encircled", width=300)
    with col3: st.image(symm, "SSIM Map for Symmetry", width=300)
    with col4: st.image(line, "Leading Lines", width=300)

    st.write(v1)
    st.write(v2)
    st.write(v3)
    st.write(v4)
    st.write(v5)
    st.write(sc)
