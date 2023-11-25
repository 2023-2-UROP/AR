import cv2
import numpy as np
# 코너 찾기, 원근 변환 코드
def get_corners(contour):
    # 윤곽선 리스트를 2차원 좌표로 재구성
    biggest_contour = contour.reshape(len(contour), 2)
    # 각 점의 x,y 좌표 합을 게산
    add_points = biggest_contour.sum(1)
    # 가장 큰 값과 가장 작은 값을 제외한 나머지 점들을 제거
    delete_index = [np.argmin(add_points), np.argmax(add_points)]
    new_points = np.delete(biggest_contour, delete_index, axis=0)

    corners = np.float32([
        biggest_contour[np.argmin(add_points)],  # 왼쪽 상단 코너
        # 모든 x좌표 중 가장 큰 값
        new_points[np.argmax(new_points[:, 0])],  # 오른쪽 상단 코너
        # 모든 x좌표 중 가장 작은 값
        new_points[np.argmin(new_points[:, 0])],  # 왼쪽 하단 코너
        biggest_contour[np.argmax(add_points)]  # 오른쪽 하단 코너
    ])
    return corners


def perspective_transform(img, shape, corners):
    # 원본 이미지에서의 모서리 4개
    pts1 = corners
    # np.float32() <- 안에 들어간 인자는 각각 왼쪽 상단, 오른쪽 상단, 왼쪽 하단, 오른쪽 하단 모서리를 의미
    pts2 = np.float32([[0, 0], [shape[0], 0], [0, shape[1]], [shape[0], shape[1]]])
    # 꼭짓점 좌표를 이용하여 변환 행렬 계산
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # 계산된 변환 행렬을 이용하여 원근 투영 변환 적용
    warp = cv2.warpPerspective(img, matrix, (shape[0], shape[1]))

    return warp