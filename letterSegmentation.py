from cv import *


def findContours(img):
    temp = CreateImage(GetSize(src),8,1) 
    Copy(img, temp)
    storage = CreateMemStorage(0)
    return FindContours(temp, storage, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE)

def getGreyScale(img):
    grey = CreateImage(GetSize(img),8,1) 
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


def similarity(im1, im2):
    # im1 is the bigger of the two, if not, swap them
    if GetSize(im1)[0] < GetSize(im2)[0]:
        im1, im2 = im2, im1
    size = GetSize(im1)
    scaledIm2 = CreateImage(size, 8, 1) 
    Resize(im2, scaledIm2)
    return 1 - ((1.0 / (255. * size[0] * size[1])) * Norm(im1, scaledIm2))
    
    

def recogniseLetterConstructor(sourceImage, letterExamples):
    def recogniseLetter(roi):
        SetImageROI(sourceImage, roi)
        probabilities = map(lambda (letter, image): (roi, letter, similarity(sourceImage, image)), letterExamples)
        ResetImageROI(sourceImage)
        probabilities.sort(reverse=True, key=lambda (roi, l, p): p)
        return probabilities[0]
    return recogniseLetter

def safeLoad(imageLocation):
    try:
        im = LoadImage(imageLocation)
        return getGreyScale(im)
    except:
        return None

def loadLetterExamples():
    alphabet = map(chr, range(ord('a'), ord('z')+1) + range(ord('A'), ord('Z')+1))
    images = map(lambda letter: (letter, safeLoad("letters/%s.png"%letter)), alphabet)
    return filter(lambda (letter, image): image != None, images)



src = LoadImage("test.png") 
thresholded = getThresholdedImage(src)
contours = findContours(thresholded)
rois = mapSequence(contours, getROI)
letterExamples = loadLetterExamples()
letterCandidates = filter(lambda (x,y,w,h): (h > 3) and (w < 20 and h < 20), rois)
letterSuggestions = map(recogniseLetterConstructor(thresholded, letterExamples), letterCandidates)

for i, (roi, letter, p) in enumerate(letterSuggestions):
    SetImageROI(src, roi)
    SaveImage("out/%s-%f.png"%(letter, p), src)


#letters = filter(lambda (roi, letter, p): p > 0.95, letterSuggestions)

#letters.sort(key=lambda ((x,y,w,h),l,p): (y+h))

#print map(lambda (roi, l, p) : l, letters)

