from preprocessing_and_contour_extraction import preprocess, extract_frame
from number_extraction_and_centering import *
from cell_processing_and_number_prediction import *
from result_visualization import *
from Solver_final import solve_wrapper

# 이미지에서 스도쿠 퍼즐의 윤곽선을 찾고, 윤곽선이 존재하는지 확인하는 함수
def check_contour(img):
    prep_img = preprocess(img)  # 이미지 전처리
    frame, contour, contour_line, thresh = extract_frame(prep_img)  # 윤곽선 추출
    contour_exist = len(contour) == 4  # 윤곽선이 4개의 모서리를 갖는지 확인
    return contour_exist, prep_img, frame, contour, contour_line, thresh  # 윤곽선 존재 여부 및 관련 데이터 반환

# 스도쿠 퍼즐의 숫자를 예측하고 퍼즐을 해결하는 함수
def predict_and_solve(img, model):
    img_nums, stats, centroids = extract_num(img)  # 숫자 추출
    centered_numbers, matrix_mask = center_numbers(img_nums, stats, centroids)  # 숫자 중심 정렬
    predicted_matrix = predict_numbers(centered_numbers, matrix_mask, model)  # 숫자 예측
    solved_matrix, solve_time = solve_wrapper(predicted_matrix.copy())  # 퍼즐 해결
    return img_nums, centered_numbers, predicted_matrix, solved_matrix, solve_time  # 해결 결과 반환

# 해결된 스도쿠 퍼즐 결과를 원본 이미지에 적용하는 함수
def apply_solution_on_image(mask, img, predicted_matrix, solved_matrix, corners):
    img_solved = display_sudoku_solution(mask, predicted_matrix, solved_matrix)  # 해결된 숫자 시각화
    inv = apply_inverse_perspective(img, img_solved, corners)  # 원근 변환 적용
    img = cv2.addWeighted(img, 1, inv, 1, 0, -1)  # 원본 이미지에 합성
    return img, img_solved  # 최종 이미지 반환
