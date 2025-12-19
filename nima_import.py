import torch
import pyiqa
import ssl
import os

def score(image_path):
    ssl._create_default_https_context = ssl._create_unverified_context
    metric = pyiqa.create_metric('nima', device=torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
    score = metric(image_path)
    print("NIMA score = ", score.item())