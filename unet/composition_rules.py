import cv2
import numpy as np
from inference import saliency_mask
image_path = ""
model_path = ""
prediction=saliency_mask(image_path, model_path)
mask = prediction.squeeze().cpu().numpy()
mask = (mask > 0.5).astype(np.uint8)*255
M = cv2.moments(mask)

if (M["m00"]!=0):
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])
else:
    cX, cY=0

image = cv2.imread(image_path)

#RULE OF THIRDS
h, w = image.shape[:2]
t1 = h * 0.1
t2 = w * 0.1
h_down = h/3
h_up = 2*h/3
w_left = w/3
w_right = 2*w/3

if (cX-w_left < t2 and cY-h_down < t1):
    verdict = "Rules of Thirds passed!"
elif (cX-w_left < t2 and cY-h_up < t1):
    verdict = "Rules of Thirds passed!"
elif (cX-w_right < t2 and cY-h_down < t1):
    verdict = "Rules of Thirds passed!"
elif (cX-w_right < t2 and cY-h_up < t1):
    verdict = "Rules of Thirds passed!"
else:
    "Rule of Thirds failed."

#GOLDEN RATIO
g_down = h/0.618
g_up = 2 * g_down
g_left = w/0.618
g_right = 2 * g_left

if (abs(cX-g_left) < t2 and abs(cY-g_down) < t1):
    verdict = "Golden Ratio passed!"
elif (abs(cX-g_left) < t2 and abs(cY-g_up < t1)):
    verdict = "Golden Ratio passed!"
elif (abs(cX-g_right) < t2 and abs(cY-g_down) < t1):
    verdict = "Golden Ratio passed!"
elif (abs(cX-g_right) < t2 and abs(cY-g_up) < t1):
    verdict = "Golden Ratio passed!"
else:
    "Golden Ratio failed."

#CENTRED OBJECT
if (abs(cX - w/2) < t2 and abs(cY - h/2) < t1):
    verdict = "Object is in the centre."
else:
    verdict = "Off-centre."

#SYMMETRY

#LEADING LINES