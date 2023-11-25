import time
from corner_detection_and_perspective_transform import *
from processing import *

output_size = (800, 600)  # 출력 이미지의 크기를 설정
model = tf.keras.models.load_model('/Users/zsu/PycharmProjects/pythonProject2/AR_CV_sudoku/model3.h5')  # 텐서플로우 모델을 불러옴

seen = False  # 스도쿠가 인식되었는지 여부
bad_read = False  # 스도쿠 읽기가 실패했는지 여부
solved = False  # 스도쿠가 해결되었는지 여부
seen_corners = 0  # 스도쿠 모서리가 인식된 시간
rectangle_counter = 0  # 탐색 사각형의 카운터

cap = cv2.VideoCapture(0)  # 카메라 캡처 시작

while cap.isOpened():  # 카메라가 열려있는 동안 반복
    success, img = cap.read()  # 카메라에서 이미지를 읽음
    img = cv2.resize(img, output_size)  # 이미지의 크기를 조정
    img_result = img.copy()  # 결과 이미지를 복사

    contour_exist, _, frame, contour, _, _ = check_contour(img_result)  # 이미지에서 윤곽선을 확인

    if contour_exist:  # 윤곽선이 존재하는 경우
        corners = get_corners(contour)  # 모서리를 찾음

        if not solved:  # 스도쿠가 해결되지 않은 경우
            color = (0, 0, 255) if bad_read else (0, 255, 0)  # 읽기 실패시 빨간색, 아니면 초록색으로 설정
            cv2.drawContours(img_result, [contour], -1, color, 2)  # 윤곽선을 그림

        if time.time() - seen_corners > 0.4:  # 지정된 시간이 지난 경우
            result = perspective_transform(frame, (450, 450), corners)  # 원근 변환을 적용
            if not seen:  # 스도쿠를 처음 보는 경우
                _, _, predicted_matrix, solved_matrix, _ = predict_and_solve(result, model)  # 스도쿠를 해결
                bad_read = np.any(solved_matrix == 0)  # 해결 실패 여부 확인
                solved = not bad_read  # 해결 여부 설정
                seen = solved  # 인식 여부 설정

            if not bad_read:  # 읽기 실패가 아닌 경우
                mask = np.zeros_like(result)  # 마스크 생성
                img_result, _ = apply_solution_on_image(mask, img_result, predicted_matrix, solved_matrix, corners)  # 해결 결과 적용

    else:  # 윤곽선이 보이지 않는 경우
        if time.time() - seen_corners > 0.2:  # 일정 시간이 지난 경우
            seen = False  # 인식 여부 초기화
            seen_corners = 0  # 인식 시간 초기화
            bad_read = False  # 읽기 실패 초기화
            solved = False  # 해결 여부 초기화
            img_result, corner_rect = draw_searching_rectangle(img_result, rectangle_counter)  # 탐색 사각형 그리기
            rectangle_counter = -1 if corner_rect > 200 else rectangle_counter + 1  # 사각형 카운터 조정

    cv2.imshow('sudoku solver', img_result)  # 결과 이미지 표시
    if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 종료
        break

cap.release()  # 카메라 해제
cv2.destroyAllWindows()  # 모든 창 닫기
