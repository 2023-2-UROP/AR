import cv2
import numpy as np

def preprocess(img):
    blurred = cv2.GaussianBlur(img, (3,3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    return gray

# def extract_frame(img):
#     # 입력 이미지와 동일한 형태로 도든 값이 0인 행열을 생성
#     zeromask = np.zeros(img.shape, np.uint8)
#
#     # 인풋 이미지에 적응형 임계값 처리를 적용해 이진화된 이미지 생성
#     threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
#     # 이진화된 이미지에서 윤곽선을 추출
#     contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     biggest_contour = []  # 가장 큰 윤곽선을 저장할 리스트
#     res = []
#     max_value = 0  # 최대 면적을 저장할 변수
#     for c in contours:
#         area = cv2.contourArea(c)  # 윤곽선의 면적을 계산
#         peri = cv2.arcLength(c, True)  # 윤곽선의 둘레를 계산
#         approx = cv2.approxPolyDP(c, 0.01 * peri, True)  # 윤곽선을 다각형으로 근사
#         # 윤곽선이 사각형 형태이며 면적이 가장 크고 40000보다 크다면 가장 큰 윤곽선으로 간주
#         if(len(approx) == 4) and (area > max_value) and (area > 40000):
#             max_value = area
#             biggest_contour = approx
#     # 가장 큰 윤곽선을 찾았다면 해당 윤곽선을 이미지에 그려 마스킹
#     if len(biggest_contour) > 0:
#         cv2.drawContours(zeromask, [biggest_contour], 0, (255), -1)
#         cv2.drawContours(zeromask, [biggest_contour], 0, (0), 2)
#         res = cv2.bitwise_and(img, zeromask)  # 원본 이미지와 마스크를 이용해 비트 연산을 통해 그리드만 추출
#     # 추출된 그리드 이미지, 가장 큰 윤곽선, 마스크 이미지, 임계값 처리된 이미지를 반환
#     return res, biggest_contour, zeromask, threshold

def extract_frame(img):
    """
    :param img: input image
    :return: image with extracted sudoku grid, biggest contour
    """
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