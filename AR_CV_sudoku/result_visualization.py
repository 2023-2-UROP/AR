import cv2
import numpy as np
# 시각화
def display_sudoku_solution(img, numbers, solved_numbers, color=(0, 255, 0)):
    cell_width = int(img.shape[1] / 9)
    cell_height = int(img.shape[0] / 9)
    # img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if len(img.shape) == 2 or (len(img.shape) > 2 and img.shape[2] == 1):
        # 그레이스케일 이미지일 경우 컬러로 변환
        img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        # 이미 컬러 이미지인 경우 변환하지 않음
        img_colored = img.copy()

    for i in range(9):
        for j in range(9):
            if numbers[j, i] == 0:
                position = (i * cell_width + int(cell_width / 4), int((j + 0.7) * cell_height))
                cv2.putText(img_colored, str(solved_numbers[j, i]), position,
                            cv2.FONT_HERSHEY_COMPLEX, 1, color, 1, cv2.LINE_AA)

    return img_colored



def apply_inverse_perspective(img, sudoku_num, corners, height=450, width=450):
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([corners[0], corners[1], corners[2], corners[3]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(sudoku_num, matrix, (img.shape[1],
                                                      img.shape[0]))
    return result


def draw_corners(img, corners):
    for corner in corners:
        cv2.circle(img, (int(corner[0]), int(corner[1])), 2, (0, 255, 0), -1)
    return img


def draw_searching_rectangle(img, counter):
    top_left = (75 + 2 * counter, 75 + 2 * counter)
    bottom_right = (img.shape[1] - 75 - 2 * counter, img.shape[0] - 75 - 2 * counter)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
    return img, top_left[0]
