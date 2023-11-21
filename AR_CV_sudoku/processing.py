from preprocessing_and_contour_extraction import preprocess, extract_frame
from number_extraction_and_centering import *
from cell_processing_and_number_prediction import *
from result_visualization import *
from Solver_final import solve_wrapper
# 전체적인 프로세스 관리 코드
def check_contour(img):
    prep_img = preprocess(img)
    frame, contour, contour_line, thresh = extract_frame(prep_img)
    contour_exist = len(contour) == 4
    return contour_exist, prep_img, frame, contour, contour_line, thresh

def predict_and_solve(img, model):
    img_nums, stats, centroids = extract_num(img)
    centered_numbers, matrix_mask = center_numbers(img_nums, stats, centroids)
    predicted_matrix = predict_numbers(centered_numbers, matrix_mask, model)
    solved_matrix, solve_time = solve_wrapper(predicted_matrix.copy())
    return img_nums, centered_numbers, predicted_matrix, solved_matrix, solve_time
def apply_solution_on_image(mask, img, predicted_matrix,solved_matrix,corners):
    img_solved = display_sudoku_solution(mask, predicted_matrix, solved_matrix)
    inv = apply_inverse_perspective(img, img_solved, corners)
    img = cv2.addWeighted(img, 1, inv, 1, 0, -1)
    return img,img_solved