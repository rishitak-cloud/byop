# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim

# image_path = "/Users/rishitakandpal/Downloads/s2.webp"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# h, w = image.shape[:2]

# h_down = h/3
# h_up = 2*h/3
# w_left = w/3
# w_right = 2*w/3
# lined = cv2.line(image, (int(w_left), 0), (int(w_left), int(h)), (255, 0, 0), 2)
# lined = cv2.line(lined, (int(w_right), 0), (int(w_right), int(h)), (255, 0, 0), 2)
# lined = cv2.line(lined, (0, int(h_down)), (int(w), int(h_down)), (255, 0, 0), 2)
# lined = cv2.line(lined, (0, int(h_up)), (int(w), int(h_up)), (255, 0, 0), 2)
# cv2.imshow("lines", lined)
# cv2.waitKey(0)

# v = np.median(image)
# sigma = 0.33
# lower = int(max(0, (1.0 - sigma) * v))
# upper = int(min(255, (1.0 + sigma) * v))
# #canny = cv2.Canny(image, lower, upper)
# kernel = np.ones((3,3), np.uint8)
# canny = cv2.GaussianBlur(image, (5,5), 0)
# canny = cv2.dilate(canny, kernel, iterations=1)

# cv2.imshow("canny", canny)
# cv2.waitKey(0)

# c = w//2
# left_part = canny[:, :c]
# right_part = canny[:, -c:]
# right = cv2.flip(right_part, 1)
# score, dif = ssim(left_part, right, full=True)
# if(score > 0.3):
#     print("Symmetry found.")
# else:
#     print("Not symmetric.")
# print(score)

import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

st.set_page_config(page_title="Image Cropper")

st.title("Simple Image Cropper")

# 1. File Uploader
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    
    # 2. The Cropper Widget
    # aspect_ratio=(1,1) forces a square. Set to None for free-form cropping.
    cropped_img = st_cropper(
        img, 
        realtime_update=True, 
        box_color='#00FF00', 
        aspect_ratio=None 
    )
    
    # 3. Display Results
    st.subheader("Result")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Original (with cropper)")
        # The cropper shows up here automatically
        
    with col2:
        st.write("Cropped Output")
        st.image(cropped_img, use_container_width=True)
        
    # 4. Optional: Download the cropped version
    if cropped_img:
        st.download_button(
            label="Download Cropped Image",
            data=uploaded_file, # Note: You'd usually save the 'cropped_img' to a buffer first
            file_name="cropped.png",
            mime="image/png"
        )
else:
    st.info("Please upload an image file in the sidebar to begin.")