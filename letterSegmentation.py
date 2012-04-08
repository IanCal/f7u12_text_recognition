from cv import *


def findContours(img):
    temp = CreateImage(GetSize(src),8,1) 
    Copy(img, temp)
    storage = CreateMemStorage(0)
    return FindContours(temp, storage, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE)

def getGreyScale(img):
    grey = CreateImage(GetSize(src),8,1) 
    CvtColor(img,grey,CV_BGR2GRAY)
    return grey

def getThresholdedImage(img):
    grey = getGreyScale(img)
    Threshold(grey, grey, 150, 255, CV_THRESH_BINARY_INV)
    return grey
    

def traverse(seq, onItem):
    while seq:
        onItem(seq)
        traverse(seq.v_next(), onItem) # Recurse on children
        seq = seq.h_next() # Next sibling

def pasteContoursFromTo(originalImage, onto):
    def curried(contour):
        return saveContour(originalImage, onto, contour)
    return curried

def saveContour(originalImage, onto, contour):
    roi = BoundingRect(list(contour))
    if (roi[2] < 20 and roi[3] < 20):
        SetImageROI(onto, roi)
        SetImageROI(originalImage, roi)
        Copy(originalImage, onto)
        ResetImageROI(onto)
        ResetImageROI(originalImage)
    

src = LoadImage("test.png") 
grey = getGreyScale(src)
thresholded = getThresholdedImage(src)
contours = findContours(thresholded)

traverse(contours, pasteContoursFromTo(thresholded, grey))

SaveImage("out.png", grey)
