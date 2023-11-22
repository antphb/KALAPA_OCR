import cv2
import numpy as np
from config import Config


def img_processing(path_img):
    image = cv2.imread(path_img)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h,w=image.shape[:2]
    image_resize = cv2.resize(gray,(int(Config.true_height/h * w),Config.true_height),interpolation=cv2.INTER_AREA)
    h,w=image_resize.shape[:2]
    try:
        img_padding = np.pad(image_resize, ((0,0),(0, Config.true_width-w)), 'median')
    except:
        img_padding = cv2.resize(gray,(Config.true_width,Config.true_height),interpolation=cv2.INTER_AREA)
    image_gau = cv2.GaussianBlur(img_padding,(5,5),1)

    thresh = cv2.adaptiveThreshold(  
                                src=image_gau,
                                maxValue=255,
                                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                thresholdType=cv2.THRESH_BINARY_INV,
                                blockSize=11,
                                C=9)
    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30,1))
    detected_lines_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    # Remove vertical lines 
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
    detected_lines_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    detected_line=cv2.bitwise_or(detected_lines_vertical,detected_lines_horizontal)
    detected_line_not=cv2.bitwise_not(detected_line)
    detected_line_and=cv2.bitwise_and(detected_line_not,thresh)
    # kernel = np.ones((1, 3), np.uint8)
    # clean_img = cv2.morphologyEx(detected_line_and, cv2.MORPH_CLOSE, kernel, iterations=2)
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    result = cv2.morphologyEx(detected_line_and, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    # result = cv2.bitwise_not(result)

    img = np.expand_dims(result, axis=2)
    img = img/255.0
    return img