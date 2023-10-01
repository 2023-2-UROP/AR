from utliy import *
import solver
# TensorFlow의 로그 메시지 레벨을 설정
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('Setting UP')

# 이미지의 높이와 너비를 설정
heightImg = 720
widthImg = 720

# 숫자 예측을 위한 모델을 초기화
model = intializePredectionModel()

# 웹 카메라로부터 비디오 입력을 받음
cap = cv2.VideoCapture(0)
while True:
    # 카메라로부터 프레임을 캡쳐
    ret, frame = cap.read()
    if not ret:
        break

    # 스도쿠 그리드를 추출
    grid = extract_sudoku_grid(frame)
    # 스도쿠 그리드가 None이 아닐 때 즉, 스도쿠 그리드를 인식했을 때 실행
    if grid is not None:
        # 추출된 그리드를 원근 투영 변환하여 스도쿠만 따로 분리합니다.
        warped = warp_perspective(frame, grid)

        # 만약 스도쿠 그리드라고 확인되면
        if is_sudoku_grid(warped):
            # 원본 프레임에 그리드 영역을 초록색으로 표시
            cv2.drawContours(frame, [grid], 0, (0, 255, 0), 2)

            # 분리한 스도쿠 영역을 이미지로 저장
            cv2.imwrite('sudoku_capture.png', warped)

            # 저장한 스도쿠 이미지를 다시 읽어옴
            pathImage = './sudoku_capture.png'
            img = cv2.imread(pathImage)
            if img is None:
                print("Failed to read sudoku_capture.png")
                continue
            img = cv2.resize(img, (widthImg, heightImg))

            # 필요한 이미지 처리를 위한 빈 이미지를 생성
            imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

            # 전처리 함수를 사용해 이미지를 이진화 및 노이즈 제거
            imgThreshold = preProcess(img)

            imgContours = img.copy()
            imgBigContour = img.copy()
            # 이진화된 이미지에서 윤곽선을 추출
            contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 추출된 윤곽선을 이미지 위에 초록색으로 그림
            cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

            # 가장 큰 윤곽선(스도쿠 그리드)를 찾음
            biggest, maxArea = biggestContour(contours)
            if biggest.size != 0:
                # 그리드의 좌표 순서를 재정렬
                biggest = reorder(biggest)

                # 원본 이미지에 상하좌우 꼭지점에 빨간점 찍음
                cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 20)

                # 그리드 영역을 원근 투영 변환하여 스도쿠 영역만 정사각형으로 분리
                # 원본 이미지에서의 모서리 4개
                pts1 = np.float32(biggest)
                # np.float32() <- 안에 들어간 인자는 각각 왼쪽 상단, 오른쪽 상단, 왼쪽 하단, 오른쪽 하단 모서리를 의미
                pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
                # 원본 이미지의 모서리 좌표(pts1)를 목표 모서리 좌표(pts2)로 매핑 시킴
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                # 원근 투영 변환 매트릭스를 이용해 원근 변환을 수행하고 그 결과를 반환
                imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
                imgDetectedDigits = imgBlank.copy()

                # 이진화된 스도쿠 영역 이미지를 가져옴
                imgWarpBinary = cv2.warpPerspective(imgThreshold, matrix, (widthImg, heightImg))

                imgSolvedDigits = imgBlank.copy()
                # 스도쿠 그리드를 9x9의 셀로 분리
                boxes = splitBoxes(imgWarpBinary)

                # 각 셀에서 숫자를 예측
                numbers = getPredection(boxes, model)

                # 예측된 숫자를 이미지에 표시
                imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
                # numbres 리스트를 numpy 배열로 변환
                # np.array() / np.asarray() 차이점 -> 복사본을 만드냐 안만드냐 차이 입력이 이미 numpy 배열일 경우에도 np.array()는 복사본을 만듦
                numbers = np.asarray(numbers)
                # 입력으로 받은 numbers 즉 스도쿠 초기 배열에서 숫자가 있으면 0 없으면 1을 나타내는 배열
                posArray = np.where(numbers > 0, 0, 1)

                # 9x9의 스도쿠 판을 생성합니다.
                board = np.array_split(numbers, 9)
                try:
                    # 스도쿠를 해결합니다.
                    resu = solver.solve(board)
                except:
                    pass
                print(board)
                # 2차원 리스트(배열)인 스도쿠(solved)를 1차원 리스트로 변환
                flatList = []
                for sublist in board:
                    for item in sublist:
                        flatList.append(item)
                # 결과적으로 flatList 에는 예측한 숫자들이 들어있음 아까 posArray는 숫자가 있으면 0 없으면 1이니까 두개 곱하면 정답인 부분들만 표현
                solvedNumbers = flatList * posArray

                # 해결된 스도쿠 숫자를 이미지에 표시
                imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)

                # 해결된 스도쿠 이미지를 원래 크기로 복원
                # 아까 원근 변환 투영해서 한거 이용해서 원본(목표) 변환된 이미지(현재) 좌표들을 이용함 대충 알겠지..?
                pts2 = np.float32(biggest)
                pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                imgInvWarpColored = img.copy()
                imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
                inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
                imgDetectedDigits = drawGrid(imgDetectedDigits)
                imgSolvedDigits = drawGrid(imgSolvedDigits)

                # 여러 이미지를 한 화면에 표시
                imageArray = ([img, imgThreshold, imgContours, imgBigContour],
                              [imgDetectedDigits, imgSolvedDigits, imgInvWarpColored, inv_perspective])
                stackedImage = stackImages(imageArray, 1)
                cv2.imshow('Stacked Images', stackedImage)

                cv2.imshow('Captured Sudoku', warped)

    cv2.imshow('Sudoku Grid Detector', frame)

    # 'q' 키를 누르면 비디오 캡쳐를 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡쳐를 종료하고 창을 닫됨
cap.release()
cv2.destroyAllWindows()

# resized_imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix,
#                                                 (frame.shape[1], frame.shape[0]))
# frame = cv2.addWeighted(resized_imgInvWarpColored, 1, frame, 0.5, 1)
