import cv2
import numpy as np

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
# def extract_num(img):
#     res = preprocess_numbers(img)
#     retval, labels, stats, centroids = cv2.connectedComponentsWithStats(res)
#     tmp = np.zeros_like(res, np.uint8)
#
#     valid_stats = []
#     valid_centroids = []
#
#     for i, (stat, centroid) in enumerate(zip(stats, centroids)):
#         if i == 0 or not is_vaild_stat(stat):
#             continue
#         tmp[labels == i] = 255
#         valid_stats.append(stat)
#         valid_centroids.append(centroid)
#     return tmp, np.array(valid_stats), np.array(valid_centroids)
#
#
# def is_vaild_stat(stat):
#     area, width, height = stat[4], stat[2], stat[3]
#     return (area > 50) and (5 <= width <= 40) and (5 <= height <= 40) and (1 <= height <= 4)


def preprocess_numbers(img):
    # 이진화
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    # 노이즈 제거를 위한 커널을 생성
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 모폴로지 연산을 통해 작은 노이즈를 제거
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    return img

# def center_numbers(img, stats, centroids):
#     # 중심에 맞춰진 숫자를 담을 그리드 이미지를 생성
#     centered_num_grid = np.zeros_like(img, np.uint8)
#     # 어떤 그리드 셀에 숫자가 있는지를 나타내는 마스크를 생성
#     matrix_mask = np.zeros((9, 9), dtype='uint8')
#     for i, number in enumerate(stats):
#         # 각 숫자에 대한 통계 정보를 얻음
#         left, top, width, height, area = stats[i]
#         # 숫자를 중심으로 배치하기 위해 새로운 좌표를 계산
#         img_left = int(((left // 50)) * 50 + ((50 - width) / 2))
#         img_top = int(((top // 50)) * 50 + ((50 - height) / 2))
#         # 각 숫자의 중심점을 가져옴
#         center = centroids[i]
#
#         # 계산된 위치에 해당 숫자를 중심으로 배치
#         centered_num_grid[img_top:img_top + height, img_left: img_left + width] = \
#             img[number[1]:number[1] + number[3], number[0]:number[0] + number[2]]
#         # 중심점을 기준으로 숫자가 위치할 그리드의 셀 위치를 계산
#         y = int(np.round((center[0] + 5) / 50, 1))
#         x = int(np.round((center[1] + 5) / 50, 1))
#         # 마스크에 해당 위치에 숫자가 있음을 표시
#         matrix_mask[x, y] = 1
#     # 중심에 맞춰진 숫자 그리드 이미지와 숫자 위치 마스크를 반환
#     return centered_num_grid, matrix_mask

def center_numbers(img, stats, centroids):
    centered_num_grid = np.zeros_like(img, np.uint8)
    matrix_mask = np.zeros((9, 9), dtype='uint8')

    for stat, centroid in zip(stats, centroids):
        img_left, img_top, width, height = calculate_centered_position(stat, centroid)
        centered_num_grid[img_top:img_top + height, img_left: img_left + width] = \
            img[stat[1]:stat[1] + height, stat[0]:stat[0] + width]
        x, y = calculate_grid_position(centroid)
        matrix_mask[y, x] = 1

    return centered_num_grid, matrix_mask

def calculate_centered_position(stat, centroid):
    left, top, width, height = stat[0], stat[1], stat[2], stat[3]
    img_left = int((left // 50) * 50 + (50 - width) / 2)
    img_top = int((top // 50) * 50 + (50 - height) / 2)
    return img_left, img_top, width, height

def calculate_grid_position(centroid):
    x = int(np.round((centroid[0] + 5) / 50, 1))
    y = int(np.round((centroid[1] + 5) / 50, 1))
    return x, y