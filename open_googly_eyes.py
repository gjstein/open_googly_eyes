import numpy as np
import cv2

# Cascade Classifiers
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')


def detect_eyes(image):
    """ Uses cascade detection to find eyes in the image 
    
    Returns: list of detected 'eyes'
    """
    
    # Make image grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize output vector
    all_eyes = []
    
    # First, detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_region, 1.2, 5)
        for (ex, ey, ew, eh) in eyes:
            # Correct for using only a region of the image
            all_eyes += [[ex+x, ey+y, ew, eh]]

    return all_eyes


cam = cv2.VideoCapture(0)
r, img = cam.read()
eyes = detect_eyes(img)

for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
