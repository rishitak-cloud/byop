import cv2
import numpy as np
from inference import mask
from skimage.metrics import structural_similarity as ssim
class analyse:
    def __init__(self, image_path, model_path):
        self.image_path = image_path
        self.model_path = model_path
        self.image = cv2.imread(image_path)
        self.h, self.w = self.image.shape[:2]
        self.prediction = mask(self.image_path, self.model_path)
        self.m = self.prediction.squeeze().cpu().numpy()
        self.m = (self.m > 0.5).astype(np.uint8)*255
        M = cv2.moments(self.m)
        if (M["m00"]!=0):
            self.cX = int(M["m10"]/M["m00"])
            self.cY = int(M["m01"]/M["m00"])
        else:
            self.cX, self.cY=0, 0

        self.t1 = self.h * 0.1
        self.t2 = self.w * 0.1

    def thirds(self):
        h_down = self.h/3
        h_up = 2*self.h/3
        w_left = self.w/3
        w_right = 2*self.w/3

        if (abs(self.cX-w_left) < self.t2 and abs(self.cY-h_down) < self.t1):
            print("Rules of Thirds passed!")
        elif (abs(self.cX-w_left) < self.t2 and abs(self.cY-h_up) < self.t1):
            print("Rules of Thirds passed!")
        elif (abs(self.cX-w_right) < self.t2 and abs(self.cY-h_down) < self.t1):
            print("Rules of Thirds passed!")
        elif (abs(self.cX-w_right) < self.t2 and abs(self.cY-h_up) < self.t1):
            print("Rules of Thirds passed!")
        else:
            print("Rule of Thirds failed.")

    def golden(self):
        g_down = self.h*0.618
        g_up = self.h*0.382
        g_left = self.w*0.618
        g_right = self.w*0.382
        if (abs(self.cX-g_left) < self.t2 and abs(self.cY-g_down) < self.t1):
            print("Golden Ratio passed!")
        elif (abs(self.cX-g_left) < self.t2 and abs(self.cY-g_up) < self.t1):
            print("Golden Ratio passed!")
        elif (abs(self.cX-g_right) < self.t2 and abs(self.cY-g_down) < self.t1):
            print("Golden Ratio passed!")
        elif (abs(self.cX-g_right) < self.t2 and abs(self.cY-g_up) < self.t1):
            print("Golden Ratio passed!")
        else:
            print("Golden Ratio failed.")

    def centre(self):
        if (abs(self.cX - self.w/2) < self.t2 and abs(self.cY - self.h/2) < self.t1):
            print("Object is in the centre.")
        else:
            print("Off-centre.")

    def symmetry(self):
        # v = np.median(image)
        # sigma = 0.33
        # lower = int(max(0, (1.0 - sigma) * v))
        # upper = int(min(255, (1.0 + sigma) * v))
        # canny = cv2.Canny(image, lower, upper)
        kernel = np.ones((3,3), np.uint8)
        canny = cv2.GaussianBlur(self.image, (5,5), 0)
        canny = cv2.dilate(canny, kernel, iterations=1)
        c = self.w//2
        left_part = canny[:, :c]
        right_part = canny[:, -c:]
        right = cv2.flip(right_part, 1)
        score, dif = ssim(left_part, right, full=True)
        if(score > 0.3):
            print("Symmetry found.")
        else:
            print("Not symmetric.")

    def lines(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        lsd = cv2.createLineSegmentDetector(0)
        dlines = lsd.detect(gray)[0]
        if dlines is not None:
            for dline in dlines:
                x1, y1, x2, y2 = map(int, dline[0])
                
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle > 90:
                    angle = 180 - angle
                if length > 0.2*self.w and (15 < angle < 75): 
                    cv2.line(self.image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.imshow("LSD Result", self.image)
        cv2.waitKey(0)