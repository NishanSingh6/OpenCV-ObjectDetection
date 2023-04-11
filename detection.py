import cv2
import numpy as np

def detect_cars(img, scale, mN, address):
    classifier = cv2.CascadeClassifier(address)
    img = cv2.resize(img, (500, 300))
    image_arr = np.array(img)

    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dilated = cv2.dilate(blur, np.ones((3, 3)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    objects = classifier.detectMultiScale(closing, scale, mN)

    count = 0

    for (x, y, w, h) in objects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1

    return img, count