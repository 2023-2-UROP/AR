from preprocessing_and_contour_extraction import preprocess, extract_frame
from number_extraction_and_centering import *
from corner_detection_and_perspective_transform import *
from cell_processing_and_number_prediction import *
from result_visualization import *
from Solver_final import solve_wrapper

def check_contour(img):
    """이미지에서 윤곽선을 체크하여 존재 여부를 반환"""
    prep_img = preprocess(img)
    frame, contour, contour_line, thresh = extract_frame(prep_img)
    contour_exist = len(contour) == 4
    return contour_exist, prep_img, frame, contour, contour_line, thresh

def predict_and_solve(img, model):
    """이미지에서 숫자를 추출하고 스도쿠를 해결"""
    img_nums, stats, centroids = extract_num(img)
    centered_numbers, matrix_mask = center_numbers(img_nums, stats, centroids)
    predicted_matrix = predict_numbers(centered_numbers, matrix_mask, model)
    solved_matrix, solve_time = solve_wrapper(predicted_matrix.copy())
    return img_nums, centered_numbers, predicted_matrix, solved_matrix, solve_time

# def apply_solution_on_image(original_img, mask, predicted_matrix, solved_matrix, corners):
#     """원본 이미지에 해결된 스도쿠를 적용."""
#     img_solved = display_sudoku_solution(mask, predicted_matrix, solved_matrix)
#     inv_perspective = apply_inverse_perspective(original_img, img_solved, corners)
#     final_img = cv2.addWeighted(original_img, 1, inv_perspective, 1, -1)
#     return final_img, img_solved
# def apply_solution_on_image(mask, original_img, predicted_matrix, solved_matrix, corners):
#     # 인버스 퍼스펙티브 변환 수행
#     inv_perspective = perspective_transform(mask, solved_matrix.shape, corners)
#     img_solved = display_sudoku_solution(mask, predicted_matrix, solved_matrix)
#     # 이미지 크기와 채널 수 출력
#     # print("Original image shape:", original_img.shape)
#     # print("Inverse perspective shape:", inv_perspective.shape)
#
#     # 이미지 크기가 다른 경우 조정
#     if original_img.shape[:2] != inv_perspective.shape[:2]:
#         inv_perspective = cv2.resize(inv_perspective, (original_img.shape[1], original_img.shape[0]))
#
#     # 이미지 채널 수가 다른 경우 조정
#     if len(original_img.shape) == 2:
#         original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
#     if len(inv_perspective.shape) == 2:
#         inv_perspective = cv2.cvtColor(inv_perspective, cv2.COLOR_GRAY2BGR)
#
#     # 조정 후 이미지 크기와 채널 수 출력
#     # print("Adjusted original image shape:", original_img.shape)
#     # print("Adjusted inverse perspective shape:", inv_perspective.shape)
#
#     # 이미지 합성
#     final_img = cv2.addWeighted(original_img, 1, inv_perspective, 1, 0)
#     return final_img, img_solved

# def apply_solution_on_image(original_img, predicted_matrix, solved_matrix, corners):
#     # 해결된 스도쿠 결과를 표시
#     solved_img = display_sudoku_solution(original_img,predicted_matrix, solved_matrix)
#
#     # 원근 변환을 적용하여 스도쿠 해결 이미지를 원본 이미지의 원근에 맞게 조정
#     inv_perspective = perspective_transform(solved_img, original_img.shape[:2], corners)
#
#     # 이미지 크기가 다른 경우 조정
#     if original_img.shape[:2] != inv_perspective.shape[:2]:
#         inv_perspective = cv2.resize(inv_perspective, (original_img.shape[0], original_img.shape[1]))
#
#     # 이미지 채널 수가 다른 경우 조정
#     if len(original_img.shape) == 2:
#         original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
#     if len(inv_perspective.shape) == 2:
#         inv_perspective = cv2.cvtColor(inv_perspective, cv2.COLOR_GRAY2BGR)
#
#     # 이미지 합성
#     final_img = cv2.addWeighted(original_img, 1, inv_perspective, 1, 0)
#     return final_img, solved_img
def apply_solution_on_image(mask, img, predicted_matrix,solved_matrix,corners):
    img_solved = display_sudoku_solution(mask, predicted_matrix, solved_matrix)
    inv = apply_inverse_perspective(img, img_solved, corners)
    img = cv2.addWeighted(img, 1, inv, 1, 0, -1)
    return img,img_solved