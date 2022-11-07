import cv2
import os
import logging
import sys
from PIL import Image

# #Crops image into a specified size defined by detect_Face
# def crop_Image(image, x, y, w, h):
#     crop_img = image[y:y+h, x:x+w]
#     #print("X: ", x , " Y: ", y, " H: ", h, " W: " , w)
#     return crop_img


#Uses pre-defined haar cascades to detect facial features
#detectMultiScale detects different object sizes and labels them
def facial_Feature(image, gray, x, y, w, h):
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 3, 2)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    nose = nose_cascade.detectMultiScale(roi_gray, 2, 3)
    for (ex,ey,ew,eh) in nose:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    mouth = mouth_cascade.detectMultiScale(roi_gray, 8, 7)
    for (ex,ey,ew,eh) in mouth:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

    return roi_color

#imports images and resizes them to a standard size
def load_images_from_folder(path):
    images = []
    for item in os.listdir(path):
        inter = cv2.INTER_AREA
        img = cv2.imread(os.path.join(path,item))
        height = 800
        dim = None
        (h, w) = img.shape[:2]
        r = height / float(h)
        dim = (int(w * r), height)
        img = cv2.resize(img, dim, interpolation = inter)
        if img is not None:
            images.append(img)
    return images


def edge_Detection(image, gray):
    img_blur = cv2.GaussianBlur(gray, (3,3), 0)
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    # cv2.imshow('Sobel X', sobelx)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel Y', sobely)
    # cv2.waitKey(0)
    # cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    # cv2.waitKey(0)


    edges = cv2.Canny(image=img_blur, threshold1=2, threshold2=65)
    return edges

def detect_Face(image):
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    # image = cv2.imread(imagePath)
    gray = cv2.cvtColor(imagePath, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),

    )
    if len(faces) == 0:

        cascPath = "haarcascade_profileface.xml"

        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(cascPath)

        # Read the image
        # image = cv2.imread(imagePath)
        gray = cv2.cvtColor(imagePath, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),

        )
        if len(faces) ==0:

            cascPath = "haarcascade_profileface.xml"

            # Create the haar cascade
            faceCascade = cv2.CascadeClassifier(cascPath)

            # Read the image
            # image = cv2.imread(imagePath)
            imageFliped = cv2.flip(imagePath,0)
            gray = cv2.cvtColor(imageFliped, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),

            )

            if len(faces) == 0:
                cascPath = "haarcascade_frontalface_default.xml"

                # Create the haar cascade
                faceCascade = cv2.CascadeClassifier(cascPath)

                # Read the image
                # image = cv2.imread(imagePath)
                imageFliped = cv2.flip(imagePath,0)
                gray = cv2.cvtColor(imageFliped, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),

                )
    return faces, gray


# logging.basicConfig(level=logging.DEBUG)
images = load_images_from_folder("images")
for image in images:
    imagePath = image
    faces, gray = detect_Face(image)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.imshow("Faces found", image)
        cv2.waitKey(0)
        cropped_image = facial_Feature(image, gray, x, y, w, h)
        edge = edge_Detection(cropped_image, gray)
        cv2.imshow("Faces found", image)
        cv2.waitKey(0)
        cv2.imshow("Faces found", edge)
        cv2.waitKey(0)






###############################
#
# # Get user supplied values
# imagePath = sys.argv[1]
# cascPath = "haarcascade_frontalface_default.xml"
#
# # Create the haar cascade
# faceCascade = cv2.CascadeClassifier(cascPath)
#
# # Read the image
# image = cv2.imread(imagePath)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Detect faces in the image
# faces = faceCascade.detectMultiScale(
#     gray,
#     scaleFactor=1.2,
#     minNeighbors=5,
#     minSize=(30, 30),
#
# )
#
# # Draw a rectangle around the faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
# cv2.imshow("Faces found", image)
# cv2.waitKey(0)
