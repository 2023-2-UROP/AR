import cv2
import numpy as np
import tensorflow as tf

def procces_cell(img):
    # 셀 이미지의 테두리를 제거
    cropped_img = img[5:img.shape[0] - 5, 5:img.shape[0] - 5]
    # 이미지의 크기를 조정
    resized = cv2.resize(cropped_img, (40, 40))
    return resized

def predict_numbers(numbers, matrix, model):
    """
    :param numbers: 추출된 숫자가 있는 이미지
    :param matrix: 빈 행렬
    :param model: 예측을 위한 모델
    :return: 예측된 숫자가 있는 행렬, 비어 있는 셀은 0
    """
    pred_list = []
    for row in range(9):
        for col in range(9):
            if matrix[row, col] == 1:
                slice = numbers[50 * row: (50 * row) + 50, 50 * col: (50 * col) + 50]
                slice = procces_cell(slice)
                slice = slice/255
                # slice = slice.reshape(1,40,40,1)
                pred_list.append(slice.reshape(1, 40, 40, 1))
                # pred_list.append(slice)
                # pred_list.append(slice.reshape(1, 40, 40, 1))

    # pred_list = np.array(pred_list)
    # print(pred_list)
    # if len(pred_list) == 0:
    #     print("pred_list is empty. Check the input data.")
    #     return np.zeros((9, 9))  # 빈 행렬 반환
    #
    # pred_list = np.array(pred_list)
    # if pred_list.ndim != 4 or pred_list.shape[1:] != (40, 40, 1):
    #     print("pred_list shape is incorrect:", pred_list.shape)
    #     return np.zeros((9, 9))
    # preds = model.predict(pred_list).argmax(axis=1)
    # 모델을 사용하여 모든 예측을 수행하고 각 예측에 대한 확률을 계산
    all_predictions = model.predict(tf.reshape(np.array(pred_list), (np.sum(matrix), 40, 40, 1)))
    probability = [np.max(prediction) for prediction in all_predictions] # 각 예측의 최대 확률
    correct = list(map(np.argmax, all_predictions)) # 각 예측 결과 중 최대값의 인덱스를 선택
    flat_matrix = matrix.flatten() # 행렬을 1차원 리스트로 평탄화
    flat_matrix[flat_matrix == 1] = correct  # flat_matrix 내에서 값이 1인 모든 위치를 correct 리스트의 값으로 바꿈

    return flat_matrix.reshape(9, 9)  # 1차원 배열인 flat_matrix를 9x9 형태의 2차원 배열로 다시 형태를 변환하여 반환
