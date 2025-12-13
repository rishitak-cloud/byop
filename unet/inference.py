import torch
import cv2
import numpy as np
from unet_parts import unet

class saliency_mask():
    def __init__(image_path, model_path):
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        model = unet(in_channels=3, num_classes=1).to(device)
        img = cv2.imread(image_path)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        input_img = cv2.resize(img, (512, 512))
        input_tensor = torch.from_numpy(input_img).float() / 255.0
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        h, w = img.shape[:2]

        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = torch.sigmoid(prediction).squeeze().cpu().numpy()
        
        resized = cv2.resize(prediction, (w,h))
        return resized