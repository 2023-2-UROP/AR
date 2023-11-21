# 카메라 실행 파일

import time as t
from corner_detection_and_perspective_transform import *
from processing import *
from Ontext import get_vars, dots

output_size = (800, 600)
model = tf.keras.models.load_model('/Users/zsu/PycharmProjects/pythonProject2/msudoku/Sudoku/model3.h5')

seen = False
bad_read = False
solved = False
seen_corners = 0
not_seen_corners = t.time() - 1
wait = 0.4
rectangle_counter = 0
time_on_corners = 0
dots_str= ""
time = ""


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    img = cv2.resize(img, output_size)
    img_result = img.copy()

    contour_exist, prep_img, frame, contour, contour_line, thresh = check_contour(img_result)

    if contour_exist:
        corners = get_corners(contour)
        not_seen_corners = 0
        out_corners_check = False
        if not solved:
            if not bad_read:
                color = (0, 0, 255) if int((10 * time_on_corners)) % 3 == 0 else (0, 255, 0)
            cv2.drawContours(img_result, [contour], -1, color, 2)
        else:
            draw_corners(img_result, corners)

        if seen_corners == 0:
            seen_corners = t.time()
        time_on_corners = t.time() - seen_corners

        if time_on_corners > wait:
            wait = 0.4
            dots_str= ''

            transformed_size = (450, 450)
            result = perspective_transform(frame, transformed_size, corners)
            if not seen:
                img_nums, centered_numbers, predicted_matrix, solved_matrix, time = predict_and_solve(result, model)
                if np.any(solved_matrix == 0):
                    bad_read = True
                    solved = False
                else:
                    bad_read = False
                    seen = True
                    solved = True
                    wait = 0.03

            if not bad_read:
                mask = np.zeros_like(result)
                img_result, img_solved = apply_solution_on_image(mask,img_result, predicted_matrix, solved_matrix, corners)

    else:
        if not_seen_corners == 0:
            not_seen_corners = t.time()
        time_out_corners = t.time() - not_seen_corners
        out_corners_check = time_out_corners > 0.2
        if out_corners_check:
            dots_str = dots(time_out_corners)
            seen = False
            seen_corners = 0
            bad_read = False
            solved = False
            wait = 0.4
            img_result, corner_rect = draw_searching_rectangle(img_result, rectangle_counter)
            if corner_rect > 200:
                rectangle_counter = -1
            rectangle_counter += 1

    text, pos, color1 = get_vars(out_corners_check,solved,bad_read,time_on_corners,seen,time)

    cv2.imshow('sudoku solver', img_result)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()