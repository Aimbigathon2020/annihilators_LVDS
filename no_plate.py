import numpy as np
import cv2
import imutils
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def noplateno(img):
    image=img
    image = imutils.resize(image, width=500)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 170, 200)
    cnts, new  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img1 = image.copy()
    cv2.drawContours(img1, cnts, -1, (0,255,0), 3)
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    NumberPlateCnt = None #we currently have no Number plate contour
    img2 = image.copy()
    cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
    count = 0
    idx =7
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print ("approx = ",approx)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour

            # Crop those contours and store it in Cropped Images folder
            x, y, w, h = cv2.boundingRect(c) #This will find out co-ord for plate
            new_img = gray[y:y + h, x:x + w] #Create new image
            cv2.imwrite('datasets/crop/' + str(idx) + '.png', new_img) #Store new image
            idx+=1

            break
    cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
    cv2.imshow("Final Image With Number Plate Detected", image)
    cv2.waitKey(0)

    Cropped_img_loc = 'datasets/crop/7.png'
    cim=cv2.imread(Cropped_img_loc)
    '''ld=cv2.CascadeClassifier('logo.xml')
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    l=ld.detectMultiScale(g,1.01,7)
    for  (x,y,w,h) in l:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("hi",img)'''
    text = pytesseract.image_to_string(np.array(img), lang='eng')
    ts=""
    for i in text:
        if((i>='a'and i<='z') or (i>='A' and i<='Z')or(i>='0'and i<='9')):
            ts+=i
    print("Number is :", ts)