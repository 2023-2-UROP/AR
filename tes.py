import solver

print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utliy import *

# pathImage = './sudoku_layout.png'
# # pathImage = './cccc.png'
pathImage = './sudoku_capture.png'

# heightImg = 450
# widthImg = 450
heightImg = 720
widthImg = 720
model = intializePredectionModel()


img = cv2.imread(pathImage)
img = cv2.resize(img,(widthImg,heightImg))
imgBlank = np.zeros((heightImg, widthImg,3),np.uint8)
imgThreshold = preProcess(img)

imgContours = img.copy()
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

biggest, maxArea = biggestContour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 20)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    # imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgWarpBinary = cv2.warpPerspective(imgThreshold, matrix, (widthImg, heightImg))

    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpBinary)
    # processed_boxes = [preProcess_onecell(cell) for cell in boxes]
    # cv2.imshow("Sample", boxes[3])
    # cv2.imwrite("sample_2.png", boxes[3])
    numbers = getPredection(boxes, model)
    # numbers = getPredection(processed_boxes, model)
    # print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    # print(posArray)

    board = np.array_split(numbers,9)
    # print(board)
    try:
        resu = solver.solve(board)
    except:
        pass
    print(board)
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList*posArray
    imgSolvedDigits = displayNumbers(imgSolvedDigits,solvedNumbers)

    pts2 = np.float32(biggest)
    pts1 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    # imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)


    imageArray = ([img, imgThreshold, imgContours, imgBigContour],
                  [imgDetectedDigits, imgSolvedDigits, imgInvWarpColored,inv_perspective])
    stackedImage = stackImages(imageArray,1)
    cv2.imshow('Stacked Images', stackedImage)

else:
    print("No Sudoku Found")
cv2.waitKey(0)

