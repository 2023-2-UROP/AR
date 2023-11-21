import cv2
import numpy as np
import tensorflow as tf
# 전처리, 숫자 예측 코드
def procces_cell(img):
    # 셀 이미지의 테두리를 제거
    cropped_img = img[5:img.shape[0] - 5, 5:img.shape[0] - 5]
    # 이미지의 크기를 조정
    resized = cv2.resize(cropped_img, (40, 40))
    return resized

def predict_numbers(numbers, matrix, model):
    pred_list = []
    for row in range(9):
        for col in range(9):
            if matrix[row, col] == 1:
                slice = numbers[50 * row: (50 * row) + 50, 50 * col: (50 * col) + 50]
                slice = procces_cell(slice)
                slice = slice/255
                pred_list.append(slice.reshape(1, 40, 40, 1))

    all_predictions = model.predict(tf.reshape(np.array(pred_list), (np.sum(matrix), 40, 40, 1)))
    probability = [np.max(prediction) for prediction in all_predictions] # 각 예측의 최대 확률
    correct = list(map(np.argmax, all_predictions)) # 각 예측 결과 중 최대값의 인덱스를 선택
    flat_matrix = matrix.flatten() # 행렬을 1차원 리스트로 평탄화
    flat_matrix[flat_matrix == 1] = correct  # flat_matrix 내에서 값이 1인 모든 위치를 correct 리스트의 값으로 바꿈

    return flat_matrix.reshape(9, 9)  # 1차원 배열인 flat_matrix를 9x9 형태의 2차원 배열로 다시 형태를 변환하여 반환
