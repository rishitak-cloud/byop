<br />
<div align="center">
<h1 align="center">FrameAgent</h1>

  <p align="center">
    A guide to improving composition in photography
    <br />
    
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

### Problem Statement
The lack of guides to help fix composition in photography while clicking photos as a beginner motivated the idea for this project. The project uses vital rules of photography (rule of thirds, golden ratio, leading lines, symmetry, centred object).
### Objective
Provide the user the option to upload a picture which will be judged on the aforementioned rules, and a crop will be suggested if required.
### Methodology
Implementation of U-Net for image segmentation and use of OpenCV to work on the image for identification and judgement. However, since the U-Net was not giving accurate results, I had to resort to using a pretrained U2-Net for presentation. I have also used a pretrained NIMA model to judge the aesthetic quality of the images.
### Final Deliverable
A website built on streamlit where the user can upload an image, it will be judged on the rules and the centroid of the mask generated is highlighted so that the user can know on what basis the decision has been made. The grids for rule of thirds and golden ratio have been shown, leading lines marked, and the SSIM map for symmetry displayed. A crop suggestion is given (I have disabled suggestion if symmetry or leading lines is followed because those are sufficient as it is). The judgements for the rules are displayed at the end, along with a NIMA score for aesthetic quality.


## Getting Started

### Prerequisites
  ```sh
  pip install torch torchvision opencv-contrib-python numpy scikit-image streamlit pyiqa rembg pillow albumentations
  ```

### Installation

 Clone the repo
   ```sh
   git clone https://github.com/rishitak-cloud/byop.git
   ```

### To run the web page
   ```sh
   streamlit run web.py
   ```


<!-- CONTACT -->
## 

Rishita Kandpal https://www.linkedin.com/in/rishita-kandpal/ https://www.kaggle.com/rishitakandpal

Project Link: [https://github.com/rishitak-cloud/byop](https://github.com/rishitak-cloud/byop)


