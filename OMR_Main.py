import cv2
import numpy as np
import Utils
path = "Answer_Sheet_(Filled).png"
# widthImg = 700
# heightImg = 800
img = cv2.imread(path)
# to re-size image
# img = cv2.resize(img, (widthImg, heightImg))
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
gradePoints = Utils.getCornerPoints(rectCon[2]) # Second biggest
# print("Biggest Contours: ",biggestContours)

if biggestContours.size != 0 and gradePoints.size !=0:
    cv2.drawContours(imgBiggestContours, biggestContours, -1, (0, 255, 0), 10)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 10)




imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny],
              [imgContours, imgBiggestContours, imgBlank, imgBlank])
imageStacked = Utils.stackImages(imageArray, 0.5)

cv2.imshow("image Stacked", imageStacked)
cv2.waitKey(0)