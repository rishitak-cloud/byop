import streamlit as st
import pandas as pd
import numpy as np
from composition_rules import analyse
from nima_import import score
from PIL import Image
#from streamlit_cropper import st_cropper
import cv2

st.set_page_config(page_title="FrameAgent")

st.title("FrameAgent", text_alignment="center")
st.space()
img = st.file_uploader(label="Choose a file", type=["jpg", "jpeg", "png", "webp"])
# model = "/Users/rishitakandpal/Downloads/model.onnx"

if img is not None:
    image = Image.open(img)
    image.save("/Users/rishitakandpal/Downloads/photo.jpg")
    path = "/Users/rishitakandpal/Downloads/photo.jpg"
    a1, a2, a3 = st.columns(3)
    with a2: st.image(img, "Uploaded file", width=300)
    st.divider()
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    thirds, v1 = analyse(path).thirds()
    golden, v2 = analyse(path).golden()
    centre, v3 = analyse(path).centre()
    symm, v4 = analyse(path).symmetry()
    line, v5 = analyse(path).lines()
    sc = score(path)
    with col1: st.image(thirds, "Grid of Thirds: The photo is divided by 4 lines, on one-thirds and two-thirds of the dimensions. Aligning the object at the points of intersection improves composition.", width=300) 
    with col2: st.image(golden, "Phi Grid for Golden Ratio: The grid is divided in the ratio of 0.382 and 0.612 in both the dimensions.", width=300)
    # with col3: st.image(centre, "Centroid encircled", width=300)
    with col3: st.image(symm, "SSIM Map for Symmetry: In an SSIM Map, white regions indicate full symmetry and the black regions indicate complete mismatch.", width=300)
    with col4: st.image(line, "Leading Lines", width=300)
    st.space()
    c1, c2, c3 = st.columns(3)
    if v5!="Lines found." and v3!="Object is in the centre.":
        st.divider()
        crop, target = analyse(path).auto_fix_image()
        with c2: st.image(crop, f"Suggested Crop {target}", width = 300)
    else:
        st.divider()
        st.write("Not suggesting crop for leading lines and symmetry.")

    st.badge(v1, color="red" if v1=="Rule of Thirds failed." else "green")
    st.badge(v2, color="red" if v2=="Golden Ratio failed." else "green")
    st.badge(v3, color="red" if v3=="Off-centre." else "green")
    st.badge(v4, color="red" if v4=="Not symmetric." else "green")
    st.badge(v5, color="red" if v5=="Lines not found." else "green")
    st.badge(sc)
    st.write("If you think that the score is low: NIMA is a measure of the aesthetic quality of a picture, which may not always match human perception. NIMA models were trained heavily on photos that have a clear focal point (a person, a flower, a bird). When it scans an image, if doesn't find a single sharp object to lock onto, or sees a texture gradient and isn't sure what it's supposed to be judging, then it defaults to a safe/average score. It also rates contrasts higher in some cases.")

