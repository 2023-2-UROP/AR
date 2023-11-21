import cv2
import numpy as np
# 스도쿠 이미지 전처리 이진화
def preprocess(img):
    blurred = cv2.GaussianBlur(img, (3,3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    return gray
def extract_frame(img):
    ramecek = np.zeros(img.shape, np.uint8)

    thresh = cv2.adaptiveThreshold(img, 255, 0, 1, 9, 5)
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    biggest_contour = []
    res = []
    max_value = 0
    for kontura in contours:
        obsah = cv2.contourArea(kontura)
        peri = cv2.arcLength(kontura, True)
        vektory = cv2.approxPolyDP(kontura, 0.01 * peri, True)
        if (len(vektory) == 4) and (obsah > max_value) and (obsah > 40000):
            max_value = obsah
            biggest_contour = vektory
    if len(biggest_contour) > 0:
        cv2.drawContours(ramecek, [biggest_contour], 0, 255, -1)
        cv2.drawContours(ramecek, [biggest_contour], 0, 0, 2)
        res = cv2.bitwise_and(img, ramecek)
    return res, biggest_contour, ramecek, thresh