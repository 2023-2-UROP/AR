import cv2
import numpy as np
import tensorflow as tf

# 모델 불러오는 함수
def intializePredectionModel():
    model = tf.keras.models.load_model('./zz_model.h5')
    return model
# 이미지 전처리 함수
def preProcess(img):
    # 그레이스케일로 변환
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # GaussianBlur로 첫번째 블러 처리 == 노이즈 제거
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    # 적응형 이진화 처리
    # imgBlur -> 이진화를 적용할 그레이스케일 이미지, 255 -> 이진화 처리 후 픽셀 값이 임계값을 초과할 경우 할당(흰색으로 할당),
    # 1 -> 적응형 이진화 방법을 지정 1은 'cv2.ADAPTIVE_THRESH_MEAN_C'를 의미 이 방법은 주변 픽셀의 평균 값에서 C값을 뺀 것을 임계값으로 사용,
    # 1 -> 1은 'cv2.THRESH_BINARY'를 의미 이 방법은 픽셀 값이 적응형 임계값보다 큰 경우 해당 픽셀을 maxValue로 설정하고, 그렇지 않으면 0으로 설정 -> 흑, 백으로 분리?,
    # 2 -> C 임계값을 계산할 때 사용하는 상수
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    # 두번째 블러 처리 == 노이즈 제거 GaussianBlur 사용하고 이진화 처리하면 숫자가 없는 셀에 경우 약간의 점? 노이즈가 존재하기 때문에 모델이 예측을 이상한 숫자로 함
    # 이런 종류의 노이즈를 salt-pepper (소금-후추) 노이즈라 함
    # medianBlur는 모든 픽셀 값들을 구하고 정렬한 후 중앙값으로 해당 픽셀들을 대체함 즉 검은 배경에 흰색 약간의 점들이 존재하면(노이즈) 이것들이 검은색으로 대체됨
    imgBlur_blur = cv2.medianBlur(imgThreshold, 5)
    return imgBlur_blur

# 점들을 재정렬하는 함수
def reorder(myPoints):
    myPoints = myPoints.reshape((4,2)) # 점의 배열을 4x2 형태로 변환
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32) # 새로 정렬할 점들을 저장할 배열 초기화

    add = myPoints.sum(1) # 각 점의 x와 y 좌표 값을 더함
    myPointsNew[0] = myPoints[np.argmin(add)] # x+y의 합이 가장 작은 점은 좌상단 점이 됨
    myPointsNew[3] = myPoints[np.argmax(add)] # x+y의 합이 가장 큰 점은 우하단 점이 됨

    diff = np.diff(myPoints, axis=1) # 각 점의 x와 y 좌표 값의 차이를 계산
    myPointsNew[1] = myPoints[np.argmin(diff)] # x-y의 차이가 가장 작은 점은 우상단 점이 됨
    myPointsNew[2] = myPoints[np.argmax(diff)] # x-y의 차이가 가장 큰 점은 좌하단 점이 됨

    return myPointsNew # 새로 정렬된 점들을 반환

# 주어진 윤곽선 중에서 가장 큰 면적을 가진 사각형의 윤곽선을 찾는 함수
def biggestContour(contours):
    biggest = np.array([]) # 가장 큰 윤곽선을 저장하기 위한 배열
    max_area = 0 # 가장 큰 윤곽선의 면적을 저장하기 위한 변수

    # 각 윤곽선에 대해서 반복
    for i in contours:
        area = cv2.contourArea(i) # 현재 윤곽선의 면적 계산

        # 면적이 50보다 큰 경우에만 아래의 로직 수행
        if area > 50:
            # 윤곽선의 둘레 계산
            peri = cv2.arcLength(i, True)

            # 윤곽선을 근사화함
            # 사각형을 표현하는 데는 4개의 꼭지점만 있으면 되지만 노이즈로나 왜곡으로 인해 수많은 점을 가진 윤곽선이 될 수도 있어서 근사화 시켜서 다각형으로 간소화
            # 인자로 True가 들어가면 근사화된 곡선은 닫힌 다각형이 됨, False면 열린 곡선으로 근사화
            approx = cv2.approxPolyDP(i, 0.02*peri, True)

            # 윤곽선의 면적이 현재 최대 면적보다 크고 꼭짓점의 수가 4인 경우에만
            # biggest와 max_area 업데이트
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    # 가장 큰 사각형 윤곽선과 그 면적 반환
    return biggest, max_area

# 81개의 셀로 분리하는 함수
def splitBoxes(img):
    rows = np.vsplit(img, 9)  # 이미지를 수직으로 9개의 부분으로 분할
    boxes = []  # 각 박스(칸)의 이미지를 저장할 리스트

    for r in rows:  # 각 행에 대하여
        cols = np.hsplit(r, 9)  # 행을 수평으로 9개의 부분으로 분할
        for box in cols:  # 각 열(박스)에 대하여
            boxes.append(box)  # boxes 리스트에 박스 이미지 추가

    return boxes  # 9x9, 총 81개의 이미지가 담긴 리스트 반환

# 주어진 셀들에 있는 숫자를 예측한느 함수
def getPredection(boxes, model):
    result = []  # 예측 결과를 저장할 리스트

    for image in boxes:  # 각 이미지 박스에 대하여
        img = np.array(image)
        img = cv2.GaussianBlur(img, (5, 5), 1)  # 이미지에 가우시안 블러 적용

        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]  # 이미지의 가장자리 부분을 제거 숫자에만 집중하게
        img = cv2.resize(img, (28, 28))  # 학습시킬 때 mnist데이터로 했는데 이 데이터 크기가 28x28라서 이미지 크기를 28x28로 조절
        img = img / 255  # 이미지의 픽셀 값을 [0,1] 범위로 정규화 딥러닝 예측에서 일반적으로 사용되는 전처리 단계 이미지의 경우 0~255까지의 값으로 구성됨
        # 그 값들을 0~1 사이로 정규화해서 보다 신경망 입력에 적합한 형태로 변환
        img = img.reshape(1, 28, 28, 1)  # 이미지의 배열의 형태를 딥러닝 인풋으로 적합한 모델로 변환

        predictions = model.predict(img)  # 모델을 사용하여 이미지에 대한 예측 수행
        classIndex= np.argmax(predictions, axis=-1)  # 가장 높은 확률 값을 갖는 클래스 인덱스 찾기
        probabilityValue = np.amax(predictions)  # 예측 확률 값 중 가장 큰 값을 찾기
        # print(classIndex, probabilityValue)  # 클래스 인덱스와 확률 값을 출력

        if probabilityValue > 0.8:  # 확률 값이 0.8보다 큰 경우
            result.append(classIndex[0])  # 결과 리스트에 클래스 인덱스 추가
        else:  # 확률 값이 0.8보다 작은 경우
            result.append(0)  # 결과 리스트에 0 추가

    return result  # 예측 결과 반환

# 이지지에 숫자를 표시하는 함수
def displayNumbers(img, numbers, color=(0, 255, 0)):
    secW = int(img.shape[1] / 9)  # 이미지의 너비를 9로 나누어 한 칸의 너비 계산
    secH = int(img.shape[0] / 9)  # 이미지의 높이를 9로 나누어 한 칸의 높이 계산

    for x in range(0, 9):  # 가로 방향으로 9칸 순회
        for y in range(0, 9):  # 세로 방향으로 9칸 순회
            if numbers[(y * 9) + x] != 0:  # 해당 위치의 숫자가 0이 아닐 경우 (스도쿠에서 빈 칸이 아님)
                # 이미지에 숫자 텍스트를 삽입
                cv2.putText(img, str(numbers[y * 9 + x]),  # 표시할 텍스트
                            (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)),  # 숫자를 중앙에 위치시키기 위해 좌표를 계산
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,  # 폰트 스타일
                            2,  # 폰트 크기
                            color,  # 텍스트 색상
                            2,  # 텍스트 두께
                            cv2.LINE_AA)  # 안티 앨리어싱 적용 -> 글자나 선을 그릴 때 안티 앨리어싱을 적용하면, 그래픽이 더 부드럽고 선명하게 나타남
    return img  # 숫자가 표시된 이미지 반환

# 주어진 이미지 위에 9x9 그리드(스도쿠 격자와 같은)를 그리는 함수
def drawGrid(img):
    secW = int(img.shape[1] / 9)  # 이미지 너비를 9로 나누어 한 칸의 너비를 계산
    secH = int(img.shape[0] / 9)  # 이미지 높이를 9로 나누어 한 칸의 높이를 계산

    for i in range(0, 9):  # 9번 반복하여 9x9 그리드를 그림
        pt1 = (0, secH * i)  # 수평선의 시작점 좌표
        pt2 = (img.shape[1], secH * i)  # 수평선의 끝점 좌표
        pt3 = (secW * i, 0)  # 수직선의 시작점 좌표
        pt4 = (secW * i, img.shape[0])  # 수직선의 끝점 좌표

        cv2.line(img, pt1, pt2, (255, 255, 0), 2)  # 이미지에 수평선 그리기 (색상은 (255,255,0) == 노란색)
        cv2.line(img, pt3, pt4, (255, 255, 0), 2)  # 이미지에 수직선 그리기 (색상은 (255,255,0) == 노란색)

    return img  # 그리드가 그려진 이미지 반환

# ------------------------------------------------------------------
def extract_sudoku_grid(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        if len(contour) >= 4:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                return approx
    return None
#  ...코드 중복...
def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_perspective(image, pts):
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def is_sudoku_grid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    lines = cv2.HoughLines(binary, 1, np.pi / 180, 150)

    if lines is None:
        return False

    vertical_lines = 0
    horizontal_lines = 0

    for rho, theta in lines[:, 0]:
        if 0 <= theta < np.pi/4 or 3*np.pi/4 <= theta < np.pi:
            vertical_lines += 1
        elif np.pi/4 <= theta < 3*np.pi/4:
            horizontal_lines += 1

    return vertical_lines >= 10 and horizontal_lines >= 10
def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0,rows):
            for y in range(0,cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0),None,scale,scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0,0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            hor_con = np.concatenate(imgArray)
            ver = hor
    return ver



