import cv2
import numpy as np
# 스도쿠 이미지 전처리 이진화
def preprocess(img):
    blurred = cv2.GaussianBlur(img, (3,3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    return gray
def extract_frame(img):
    # 이미지에서 가장 큰 사각형 영역 추출하는 함수 정의
    mask = np.zeros(img.shape, np.uint8)  # 동일한 크기의 검은색 마스크 생성

    threshold_img = cv2.adaptiveThreshold(img, 255, 0, 1, 9, 5)
    # 이미지에 적응형 임계값 적용하여 이진화

    contours, hier = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 이진화된 이미지에서 윤곽선 찾기

    largest_contour = []  # 가장 큰 윤곽선을 저장할 리스트 초기화
    largest_rectangle = []  # 최종 사각형 영역을 저장할 리스트 초기화
    max_area = 0  # 최대 면적 값을 저장할 변수 초기화
    for contour in contours:  # 모든 윤곽선에 대해 반복
        area = cv2.contourArea(contour)  # 윤곽선 면적 계산
        perimeter = cv2.arcLength(contour, True)  # 윤곽선 둘레 길이 계산
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        # 윤곽선을 근사하여 간단한 형태로 변환

        # 사각형 윤곽선이고, 면적이 최대이며, 40000보다 큰 경우
        if len(approx) == 4 and area > max_area and area > 40000:
            max_area = area  # 최대 면적 갱신
            largest_contour = approx  # 가장 큰 윤곽선 갱신

    if len(largest_contour) > 0:  # 가장 큰 윤곽선이 존재하는 경우
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        # 가장 큰 윤곽선 내부를 흰색으로 채움
        cv2.drawContours(mask, [largest_contour], 0, 0, 2)
        # 가장 큰 윤곽선 외곽선을 검은색으로 그림
        largest_rectangle = cv2.bitwise_and(img, mask)
        # 원본 이미지와 가장 큰 윤곽선을 AND 연산하여 결과 이미지 생성

    return largest_rectangle, largest_contour, mask, threshold_img
    # 결과 이미지, 가장 큰 윤곽선, 윤곽선 마스크 이미지, 임계값 적용 이미지 반환
