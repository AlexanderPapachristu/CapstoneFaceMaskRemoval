import cv2
import os
import logging
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
import skimage
model=load_model("./model2-010.model")

# #Crops image into a specified size defined by detect_Face
# def crop_Image(image, x, y, w, h):
#     crop_img = image[y:y+h, x:x+w]
#     #print("X: ", x , " Y: ", y, " H: ", h, " W: " , w)
#     return crop_img


#Uses pre-defined haar cascades to detect facial features
#detectMultiScale detects different object sizes and labels them
#Region of interest functionality for gray and color spaces
def facial_Feature(image, gray, x, y, w, h):
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    # nose = nose_cascade.detectMultiScale(roi_gray, 2, 3)
    # for (ex,ey,ew,eh) in nose:
    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    # mouth = mouth_cascade.detectMultiScale(roi_gray, 8, 7)
    # for (ex,ey,ew,eh) in mouth:
    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

    return roi_color, roi_gray

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

# 
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




# def detect_Face(image):
#     cascPath = "haarcascade_frontalface_default.xml"

#     # Create the haar cascade
#     faceCascade = cv2.CascadeClassifier(cascPath)

#     # Read the image
#     # image = cv2.imread(imagePath)
#     gray = cv2.cvtColor(imagePath, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the image
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),

#     )
#     if len(faces) == 0:

#         cascPath = "haarcascade_profileface.xml"

#         # Create the haar cascade
#         faceCascade = cv2.CascadeClassifier(cascPath)

#         # Read the image
#         # image = cv2.imread(imagePath)
#         gray = cv2.cvtColor(imagePath, cv2.COLOR_BGR2GRAY)

#         # Detect faces in the image
#         faces = faceCascade.detectMultiScale(
#             gray,
#             scaleFactor=1.3,
#             minNeighbors=5,
#             minSize=(30, 30),

#         )
#         if len(faces) ==0:

#             cascPath = "haarcascade_profileface.xml"

#             # Create the haar cascade
#             faceCascade = cv2.CascadeClassifier(cascPath)

#             # Read the image
#             # image = cv2.imread(imagePath)
#             imageFliped = cv2.flip(imagePath,0)
#             gray = cv2.cvtColor(imageFliped, cv2.COLOR_BGR2GRAY)

#             # Detect faces in the image
#             faces = faceCascade.detectMultiScale(
#                 gray,
#                 scaleFactor=1.1,
#                 minNeighbors=5,
#                 minSize=(30, 30),

#             )

#             if len(faces) == 0:
#                 cascPath = "haarcascade_frontalface_default.xml"

#                 # Create the haar cascade
#                 faceCascade = cv2.CascadeClassifier(cascPath)

#                 # Read the image
#                 # image = cv2.imread(imagePath)
#                 imageFliped = cv2.flip(imagePath,0)
#                 gray = cv2.cvtColor(imageFliped, cv2.COLOR_BGR2GRAY)

#                 # Detect faces in the image
#                 faces = faceCascade.detectMultiScale(
#                     gray,
#                     scaleFactor=1.1,
#                     minNeighbors=5,
#                     minSize=(30, 30),

#                 )          
#     return faces

def detect_shape(img):
    # read image through command line
    # img = cv2.imread(args["ipimage"])

    # convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(gray_image,127,255,0)

    # find contours in the binary image
    contours, hierarchy= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
    # calculate moments for each contour
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # display the image
        cv2.imshow("Image", img)
        cv2.waitKey(0)
    return img



# MAIN FUNCTION, LORD FORGIVE ME FOR WHAT I'M ABOUT TO CODE
images = load_images_from_folder("images")
labels_dict={0:'without mask',1:'mask'}
color_dict={0:(0,0,255),1:(0,255,0)}
size = 4
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# while True:
for image in images:
    imagePath = image
    # faces = detect_Face(image)
    # (rval, image) = webcam.read()

    mini = cv2.resize(image, (image.shape[1] // size, image.shape[0] // size))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Draw a rectangle around the faces
    for f in faces:
        (x, y, w, h) = [v * size for v in f]
        face_img = image[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        label=np.argmax(result,axis=1)[0]

        cv2.rectangle(image,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(image,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(image, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)



        if labels_dict[label] == "mask":
            cropped_image, cropped_gray = facial_Feature(image, gray, x, y, w, h)
            edged_image = edge_Detection(cropped_image, cropped_gray)
            # shaped_image = detect_shape(cropped_image)
            cv2.imshow("Image", cropped_image)
            cv2.waitKey(0)
        

    



