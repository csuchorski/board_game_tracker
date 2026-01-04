import cv2

ORB = cv2.ORB.create(nfeatures=5000)
BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
