import cv2
import numpy as np
import Utils
path = "Answer_Sheet_(Filled).png"
widthImg = 700
heightImg = 700
questions = 5
choices = 5
img = cv2.imread(path)
# to re-size image
img = cv2.resize(img, (widthImg, heightImg))
# PROCESSING
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
# FIND RECTANGLES
rectCon= Utils.rectCountour(contours)
biggestContours = Utils.getCornerPoints(rectCon[0]) # biggest
print(biggestContours.shape)
gradePoints = Utils.getCornerPoints(rectCon[2]) # Second biggest
# print("Biggest Contours: ",biggestContours)

if biggestContours.size != 0 and gradePoints.size !=0:
    cv2.drawContours(imgBiggestContours, biggestContours, -1, (0, 255, 0), 10)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 10)

    biggestContours = Utils.reorder(biggestContours)
    gradePoints = Utils.reorder(gradePoints)

    pt1 = np.float32(biggestContours)
    pt2 = np.float32([[0, 0],[widthImg,0], [0, heightImg], [widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored= cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0, 0],[325,0], [0, 150], [325,150]])
    matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgGrayDisplay= cv2.warpPerspective(img, matrixG, (325, 150))
    #cv2.imshow("Grade", imgGrayDisplay)

    # APPLY THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imageThresh = cv2.threshold(imgWarpGray, 170, 225, cv2.THRESH_BINARY_INV)[1]

    boxes = Utils.splitBoxes(imageThresh)
    # cv2.imshow("Boxes", boxes[0])
    # print(cv2.countNonZero(boxes[0]), cv2.countNonZero(boxes[1]))
    myPixelVal = np.zeros((questions, choices))
    countC=0
    countR=0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC +=1
        if(countC == choices):countR +=1; countC=0
    print(myPixelVal)

imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny],
              [imgContours, imgBiggestContours, imgWarpColored, imageThresh])
imageStacked = Utils.stackImages(imageArray, 0.5)

cv2.imshow("image Stacked", imageStacked)
cv2.waitKey(0)