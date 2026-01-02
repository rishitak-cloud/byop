import cv2
import numpy as np
from rembg import remove, new_session

def mask(image_path):
    img = cv2.imread(image_path)
    model_name = "u2net" 
    session = new_session(model_name)
    result_rgba = remove(img, session=session)
    mask_only = result_rgba[:, :, 3]
    mask_float = mask_only.astype(np.float32) / 255.0
    
    return mask_float