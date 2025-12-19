import cv2
import numpy as np
from inference import saliency_mask
from skimage.metrics import structural_similarity as ssim

image_path = ""
model_path = ""
img = cv2.imread(image_path)
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
v = np.median(img)
sigma = 0.33
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
canny = cv2.Canny(img, lower, upper)
c = w//2
left_part = canny[:, :c]
right_part = canny[:, -c:]
right = cv2.flip(right_part, 1)
score, dif = ssim(left_part, right, full=True)
if(score > 0.9):
    verdict = "High symmetry."
elif(0.8 < score <= 0.9):
    verdict = "Average symmetry."
else:
    verdict = "Not symmetric."

#LEADING LINES
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lsd = cv2.createLineSegmentDetector(0)
dlines = lsd.detect(gray)[0]
h, w = img.shape[:2]

filtered_lines = []
if dlines is not None:
    for dline in dlines:
        x1, y1, x2, y2 = map(int, dline[0])
        
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle > 90:
            angle = 180 - angle
        if length > 0.2*w and (15 < angle < 75): 
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imshow("LSD Result", img)
cv2.waitKey(0)