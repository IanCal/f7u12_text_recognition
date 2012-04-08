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
    

def mapSequence(seq, onItem):
    acc = []
    while seq:
        acc.append(onItem(seq))
        acc.extend(mapSequence(seq.v_next(), onItem))
        seq = seq.h_next() # Next sibling
    return acc

def getROI(contour):
    return BoundingRect(list(contour))

def recogniseLetterConstructor(sourceImage, letterExamples):
    def recogniseLetter(roi):
        SetImageROI(sourceImage, roi)
        probabilities = map(lambda (letter, image): (letter, similarity(sourceImage, image)), letterExamples)
        probabilities.sort(reverse=True, key=itemgetter(2))
        return probabilities[0]
    return recogniseLetter

def loadLetterExamples():
    alphabet = map(chr, range(ord('a'), ord('z')+1) + range(ord('A'), ord('Z')+1))
    return map(lambda letter: (letter, LoadImage("letters/%s.png"%letter)), alphabet)

src = LoadImage("test.png") 
thresholded = getThresholdedImage(src)
contours = findContours(thresholded)
rois = mapSequence(contours, getROI)
letterExamples = loadLetterExamples()
letterCandidates = filter(lambda (x,y,w,h): (h > 3) and (w < 20 and h < 20), rois)
letterSuggestions = map(recogniseLetterConstructor(thresholded, letterExamples), letterCandidates)
letters = filter(lambda (roi, letter, p): p > 0.5, letterSuggestions)

