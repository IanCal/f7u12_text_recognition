from cv import *


src = LoadImageM("test.png") 
grey = CreateImage(GetSize(src),8,1) 
backup = CreateImage(GetSize(src),8,1) 
CvtColor(src,grey,CV_BGR2GRAY)

Threshold(grey, grey, 150, 255, CV_THRESH_BINARY_INV)
Copy(grey, backup)

storage = CreateMemStorage(0)
contours = FindContours(grey, storage, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE)

def boundSize(contour):
    bound_rect = BoundingRect(list(contour))
    return (bound_rect[2], bound_rect[3])

    

def traverse(seq, onItem):
    while seq:
        onItem(seq)
        traverse(seq.v_next(), onItem) # Recurse on children
        seq = seq.h_next() # Next sibling


def saveContour(contour):
    roi = BoundingRect(list(contour))
    cropped = CreateImage((roi[2], roi[3]), 8, 1)
    src_region = GetSubRect(backup, roi )
    Copy(src_region, cropped)
    SaveImage("lots/%d-%d-%d-%d.png"%(roi[2], roi[3], roi[0], roi[1]), cropped) 
    


traverse(contours, saveContour)

#    bound_rect = BoundingRect(list(contour))

    #centers.append(bound_rect[0] + bound_rect[2] / 2, bound_rect[1] + bound_rect[3] / 2)


