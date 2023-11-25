import cv2
import numpy as np
# 숫자를 추출하고 가운데 정렬
def extract_num(img):
    # 숫자를 전처리하는 함수를 호출
    result = preprocess_numbers(img)
    # 연결된 컴포넌트를 분석하여 레이블링, 통계 및 중심점을 반환
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(result)
    # 시각화를 위한 빈 이미지를 생성
    viz = np.zeros_like(result, np.uint8)

    centroidy = []  # 중심점을 저장할 리스트를 초기화
    stats_numbers = []  # 통계를 저장할 리스트를 초기화

    # 통계 배열을 순회하면서 조건에 맞는 객체(숫자)를 찾음
    for i, stat in enumerate(stats):
        if i == 0:  # 레이블 0은 배경을 나타내므로 건너뜀
            continue
        # 면적이 50 이상이고, 너비와 높이가 5에서 40 사이, 비율이 1:1부터 1:4 사이인 컴포넌트를 숫자로 간주
        if stat[4] > 50 and stat[2] in range(5,40) and stat[3] in range(5,40) and stat[0] > 0 and stat[
            1] > 0 and (int(stat[3] / stat[2])) in range(1,5):
            viz[labels == i] = 255  # 해당 컴포넌트를 시각화 이미지에 표시
            centroidy.append(centroids[i])  # 중심점을 리스트에 추가
            stats_numbers.append(stat)  # 통계를 리스트에 추가

    # 리스트를 NumPy 배열로 변환
    stats_numbers = np.array(stats_numbers)
    centroidy = np.array(centroidy)
    return viz, stats_numbers, centroidy  # 결과 이미지, 통계 배열, 중심점 배열을 반환
def preprocess_numbers(img):
    # 이진화
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    # 노이즈 제거를 위한 커널을 생성
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 모폴로지 연산을 통해 작은 노이즈를 제거
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    return img
def center_numbers(img, stats, centroids):
    # 이미지에서 추출된 숫자를 가운데 정렬하고, 정렬된 숫자의 격자 위치를 표시하는 함수
    centered_num_grid = np.zeros_like(img, np.uint8)  # 숫자를 정렬할 빈 이미지 생성
    matrix_mask = np.zeros((9, 9), dtype='uint8')  # 숫자의 격자 위치를 표시할 행렬 생성

    for stat, centroid in zip(stats, centroids):  # 각 숫자의 통계와 중심점에 대해 반복
        img_left, img_top, width, height = calculate_centered_position(stat, centroid)  # 숫자를 정렬할 위치 계산
        centered_num_grid[img_top:img_top + height, img_left: img_left + width] = \
            img[stat[1]:stat[1] + height, stat[0]:stat[0] + width]  # 추출된 숫자를 새 위치에 복사
        x, y = calculate_grid_position(centroid)  # 숫자의 격자 위치 계산
        matrix_mask[y, x] = 1  # 격자 위치에 숫자 존재 표시

    return centered_num_grid, matrix_mask  # 정렬된 숫자 이미지와 격자 위치 행렬 반환

def calculate_centered_position(stat, centroid):
    # 추출된 숫자를 정렬할 위치를 계산하는 함수
    left, top, width, height = stat[0], stat[1], stat[2], stat[3]  # 숫자의 위치와 크기 정보
    img_left = int((left // 50) * 50 + (50 - width) / 2)  # 숫자를 격자 가운데 정렬할 왼쪽 위치 계산
    img_top = int((top // 50) * 50 + (50 - height) / 2)  # 숫자를 격자 가운데 정렬할 상단 위치 계산
    return img_left, img_top, width, height  # 계산된 위치와 크기 반환

def calculate_grid_position(centroid):
    # 숫자의 격자 위치를 계산하는 함수
    x = int(np.round((centroid[0] + 5) / 50, 1))  # 격자 내에서 숫자의 x 좌표 계산
    y = int(np.round((centroid[1] + 5) / 50, 1))  # 격자 내에서 숫자의 y 좌표 계산
    return x, y  # 계산된 격자 위치 반환
