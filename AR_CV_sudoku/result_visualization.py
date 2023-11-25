import cv2
import numpy as np

# 스도쿠 퍼즐 이미지에 해결된 숫자를 표시하는 함수
def display_sudoku_solution(img, numbers, solved_numbers, color=(0, 255, 0)):
    cell_width = int(img.shape[1] / 9)  # 셀의 너비 계산
    cell_height = int(img.shape[0] / 9)  # 셀의 높이 계산

    if len(img.shape) == 2 or (len(img.shape) > 2 and img.shape[2] == 1):
        img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 그레이스케일 이미지를 컬러로 변환
    else:
        img_colored = img.copy()  # 이미 컬러 이미지인 경우 복사

    for i in range(9):
        for j in range(9):
            if numbers[j, i] == 0:  # 빈 셀에 숫자를 표시
                position = (i * cell_width + int(cell_width / 4), int((j + 0.7) * cell_height))
                cv2.putText(img_colored, str(solved_numbers[j, i]), position,
                            cv2.FONT_HERSHEY_COMPLEX, 1, color, 1, cv2.LINE_AA)

    return img_colored  # 숫자가 표시된 이미지 반환

# 원근 변환된 이미지를 원본 이미지 크기로 되돌리는 함수
def apply_inverse_perspective(img, sudoku_num, corners, height=450, width=450):
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # 원근 변환된 이미지의 모서리 좌표
    pts2 = np.float32([corners[0], corners[1], corners[2], corners[3]])  # 원본 이미지의 모서리 좌표
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # 변환 행렬 계산
    result = cv2.warpPerspective(sudoku_num, matrix, (img.shape[1], img.shape[0]))  # 변환 적용

    return result  # 변환된 이미지 반환

# 스도쿠 퍼즐의 모서리에 원을 그리는 함수
def draw_corners(img, corners):
    for corner in corners:
        cv2.circle(img, (int(corner[0]), int(corner[1])), 2, (0, 255, 0), -1)  # 모서리 좌표에 초록색 원을 그림

    return img  # 원이 그려진 이미지 반환

# 스도쿠 퍼즐을 찾기 위한 사각형을 그리는 함수
def draw_searching_rectangle(img, counter):
    top_left = (75 + 2 * counter, 75 + 2 * counter)  # 사각형의 왼쪽 상단 좌표
    bottom_right = (img.shape[1] - 75 - 2 * counter, img.shape[0] - 75 - 2 * counter)  # 사각형의 오른쪽 하단 좌표
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)  # 빨간색 사각형을 그림

    return img, top_left[0]  # 사각형이 그려진 이미지와 사각형의 왼쪽 상단 x좌표 반환
