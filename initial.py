import cv2
import os
import logging
import sys
from PIL import Image

def load_images_from_folder(path):
    images = []
    for item in os.listdir(path):
        # baseheight = 800
        # img = Image.open(path+'\\'+item)
        # f, e = os.path.splitext(path+item)
        # hpercent = (baseheight / float(img.size[1]))
        # wsize = int((float(img.size[0]) * float(hpercent)))
        # img = img.resize((wsize, baseheight), Image.ANTIALIAS)
        # # imResize = im.resize((200,200), Image.ANTIALIAS)
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

# logging.basicConfig(level=logging.DEBUG)
images = load_images_from_folder("images")
for image in images:
    imagePath = image
    cascPath = "haarcascade_frontalface_default.xml"

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
    if(faces is None):
        cascPath = "haarcascade_frontalface_default.xml"

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
        if(faces is None):
                    cascPath = "haarcascade_frontalface_default.xml"

                    # Create the haar cascade
                    faceCascade = cv2.CascadeClassifier(cascPath)

                    # Read the image
                    # image = cv2.imread(imagePath)
                    imageFliped = cv2.flip(imagePath)
                    gray = cv2.cvtColor(imageFliped, cv2.COLOR_BGR2GRAY)

                    # Detect faces in the image
                    faces = faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.3,
                        minNeighbors=5,
                        minSize=(30, 30),

                    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
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
