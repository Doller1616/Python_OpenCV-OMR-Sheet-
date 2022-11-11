import cv2
import numpy as np
import Utils
path = "Answer_Sheet_(Filled).png"
widthImg = 700
heightImg = 700
questions = 5
choices = 5
ans = [0, 2, 3, 1, 3]
webcamFeed = True
cameraNo = 0
cap = cv2.VideoCapture(cameraNo)
cap.set(10, 150)

while True:
    if webcamFeed: success, img = cap.read()
    else: img = cv2.imread(path)

    # PREPROCESSING
    img = cv2.resize(img, (widthImg, heightImg))
    imgFinal = img.copy();
    # PROCESSING
    imgContours = img.copy()
    imgBiggestContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    try:
        # FINDING ALL CONTOURS
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

            # GETTING NON ZERO PIXELS VALUES
            myPixelVal = np.zeros((questions, choices))
            countC=0
            countR=0
            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC +=1
                if(countC == choices):countR +=1; countC=0
            print(myPixelVal)

            # FINDING INDEX VALUES OF THE MARKINGS
            myIndex = [];
            for x in range (0, questions):
                arr = myPixelVal[x]
                # print("arr", arr)
                myIndexVal = np.where(arr==np.amax(arr));
                # print(myIndexVal[0])
                myIndex.append(myIndexVal[0][0])
            print(myIndex)

            # GRADING
            grading=[]
            for x in range (0, questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else: grading.append(0)
            # print(grading)
            score = (sum(grading)/questions) * 100 # Final GRADE
            # print(score)

            # DISPLAYING ANSWER
            imgResult = imgWarpColored.copy()
            imgResult = Utils.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
            imgRawDrawing = np.zeros_like(imgWarpColored)
            imgRawDrawing = Utils.showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)
            invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

            imgRawGrade = np.zeros_like(imgGrayDisplay)
            cv2.putText(imgRawGrade, str(int(score))+"%", (70,100), cv2.FONT_HERSHEY_COMPLEX, 3,(0,225, 225), 3)
            invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0);
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0);

        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                      [imgContours, imgBiggestContours, imgWarpColored, imageThresh],
                      [imgResult, imgRawDrawing,imgInvWarp,imgFinal])
    except:
        imgBlank = np.zeros_like(img)
        imageArray = ([imgBlank, imgBlank, imgBlank, imgBlank],
                  [imgBlank, imgBlank, imgBlank, imgBlank],
                  [imgBlank, imgBlank, imgBlank,imgBlank])


    labels =[["Original","Gray", "Blur", "Canny", ""],
              ["Contours", "Biggest Con", "Wrap", "Threshhold"],
              ["Result", "Raw Drawing", "Inv Wrap", "Final"]]
    imageStacked = Utils.stackImages(imageArray, 0.4, labels)

    cv2.imshow("Final Result", imgFinal)
    cv2.imshow("image Stacked", imageStacked)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("FinalResult.jpg", imgFinal)
        cv2.waitKey(300)