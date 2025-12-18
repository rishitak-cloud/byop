import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def hough():
    img_path = "/Users/rishitakandpal/Downloads/l1.jpeg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    blur = cv2.GaussianBlur(img, (9,9), 0)

    v = np.median(img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    canny = cv2.Canny(blur, lower, upper)

    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(canny, kernel, iterations=1)
    #cv2.imshow("canny", canny)

    h, w = img.shape[:2]
    dim = min(h,w)
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, int(0.2*dim), int(0.2*dim), int(0.1*dim))

    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angle = abs(angle)
            
            if angle > 90:
                angle = 180 - angle
            
            if 15 < angle < 75:
                filtered_lines.append(line)

    # if(len(filtered_lines) < 15):
    #     kernel = np.ones((3,3), np.uint8)
    #     dilated_edges = cv2.dilate(canny, kernel, iterations=1)
    #     cv2.imshow("dilated", dilated_edges)
    #     lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, int(0.2*dim), int(0.2*dim), int(0.05*dim))

    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
 
    cv2.imshow("some", img_bgr)
    cv2.waitKey()
    return 0
if __name__=="__main__":
    hough()