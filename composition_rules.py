import cv2
import numpy as np
from inference import mask
from skimage.metrics import structural_similarity as ssim
class analyse:
    def __init__(self, image_path, model_path):
        self.image_path = image_path
        self.model_path = model_path
        self.image = cv2.imread(image_path)
        self.h, self.w = (self.image).shape[:2]
        self.prediction = mask(self.image_path, self.model_path)
        #self.m = self.prediction.squeeze().cpu().numpy()
        self.m = self.prediction.squeeze()
        self.m = (self.m > 0.5).astype(np.uint8)*255
        M = cv2.moments(self.m)
        if (M["m00"]!=0):
            self.cX = int(M["m10"]/M["m00"])
            self.cY = int(M["m01"]/M["m00"])
        else:
            self.cX, self.cY=0, 0

        self.t1 = self.h * 0.05
        self.t2 = self.w * 0.05

    def thirds(self):
        h_down = self.h/3
        h_up = 2*self.h/3
        w_left = self.w/3
        w_right = 2*self.w/3

        if (abs(self.cX-w_left) < self.t2 and abs(self.cY-h_down) < self.t1):
            v = "Rules of Thirds passed!"
        elif (abs(self.cX-w_left) < self.t2 and abs(self.cY-h_up) < self.t1):
            v = "Rules of Thirds passed!"
        elif (abs(self.cX-w_right) < self.t2 and abs(self.cY-h_down) < self.t1):
            v = "Rules of Thirds passed!"
        elif (abs(self.cX-w_right) < self.t2 and abs(self.cY-h_up) < self.t1):
            v = "Rules of Thirds passed!"
        else:
            v = "Rule of Thirds failed."
        lined = self.image.copy()
        lined = cv2.line(lined, (int(w_left), 0), (int(w_left), int(self.h)), (255, 0, 0), 2)
        lined = cv2.line(lined, (int(w_right), 0), (int(w_right), int(self.h)), (255, 0, 0), 2)
        lined = cv2.line(lined, (0, int(h_down)), (int(self.w), int(h_down)), (255, 0, 0), 2)
        lined = cv2.line(lined, (0, int(h_up)), (int(self.w), int(h_up)), (255, 0, 0), 2)
        circled=cv2.circle(lined, center=(int(self.cX), int(self.cY)), radius=10, color=(0,255,0), thickness=50)
        # cv2.imshow("circled", lined)
        # cv2.waitKey(0)
        return circled, v

    def golden(self):
        g_down = self.h*0.618
        g_up = self.h*0.382
        g_left = self.w*0.618
        g_right = self.w*0.382
        if (abs(self.cX-g_left) < self.t2 and abs(self.cY-g_down) < self.t1):
            v = "Golden Ratio passed!"
        elif (abs(self.cX-g_left) < self.t2 and abs(self.cY-g_up) < self.t1):
            v = "Golden Ratio passed!"
        elif (abs(self.cX-g_right) < self.t2 and abs(self.cY-g_down) < self.t1):
            v = "Golden Ratio passed!"
        elif (abs(self.cX-g_right) < self.t2 and abs(self.cY-g_up) < self.t1):
            v = "Golden Ratio passed!"
        else:
            v = "Golden Ratio failed."
        lined = self.image.copy()
        lined = cv2.line(lined, (int(g_left), 0), (int(g_left), int(self.h)), (255, 0, 0), 2)
        lined = cv2.line(lined, (int(g_right), 0), (int(g_right), int(self.h)), (255, 0, 0), 2)
        lined = cv2.line(lined, (0, int(g_down)), (int(self.w), int(g_down)), (255, 0, 0), 2)
        lined = cv2.line(lined, (0, int(g_up)), (int(self.w), int(g_up)), (255, 0, 0), 2)
        circled=cv2.circle(lined, center=(int(self.cX), int(self.cY)), radius=10, color=(0,255,0), thickness=50)
        return circled, v

    def centre(self):
        if (abs(self.cX - self.w/2) < self.t2 and abs(self.cY - self.h/2) < self.t1):
            v = "Object is in the centre."
        else:
            v = "Off-centre."
        circled = self.image.copy()
        circled=cv2.circle(circled, center=(self.cX, self.cY), radius=10, color=(0,255,0), thickness=50)
        return circled, v

    def symmetry(self):
        # v = np.median(image)
        # sigma = 0.33
        # lower = int(max(0, (1.0 - sigma) * v))
        # upper = int(min(255, (1.0 + sigma) * v))
        # canny = cv2.Canny(image, lower, upper)
        kernel = np.ones((3,3), np.uint8)
        canny = self.image.copy()
        canny = cv2.GaussianBlur(canny, (5,5), 0)
        canny = cv2.dilate(canny, kernel, iterations=1)
        c = self.w//2
        left_part = canny[:, :c]
        right_part = canny[:, -c:]
        right = cv2.flip(right_part, 1)

        w_size = 7 if left_part.shape[1] >= 7 and left_part.shape[0] >= 7 else 3
        score, dif = ssim(left_part, right, full=True, win_size=w_size, data_range=255, channel_axis=2)

        if(score > 0.3):
            v = f"Symmetry found, score = {score}"
        else:
            v = "Not symmetric."
        dif_visual = (dif * 255).astype(np.uint8)
        return dif_visual, v

    def lines(self):
        gray = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2GRAY)
        lsd = cv2.createLineSegmentDetector(0)
        dlines = lsd.detect(gray)[0]
        v = "Lines not found."
        if dlines is not None:
            for dline in dlines:
                x1, y1, x2, y2 = map(int, dline[0])
                
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle > 90:
                    angle = 180 - angle
                if length > 0.2*self.w and (15 < angle < 75): 
                    cv2.line(self.image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    v = "Lines found."

        # cv2.imshow("LSD Result", self.image)
        # cv2.waitKey(0)
        
        return self.image, v