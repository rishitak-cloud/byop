import torch
import pyiqa
import ssl
import os

def score(image_path):
    ssl._create_default_https_context = ssl._create_unverified_context
    metric = pyiqa.create_metric('nima', device="cpu")
    score = metric(image_path)
    v= f"NIMA score = {score.item()}"
    return v