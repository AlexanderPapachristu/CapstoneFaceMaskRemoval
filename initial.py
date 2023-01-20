from contextlib import nullcontext
import cv2
import os
import logging
import sys
import ctypes
import numpy as np
from PIL import Image

from keras.models import load_model
import skimage
from inpaint_model import InpaintCAModel
import neuralgym as ng
import tensorflow as tf
model=load_model("./model2-005.model")

#Uses pre-defined haar cascades to detect facial features
#detectMultiScale detects different object sizes and labels them
#Region of interest functionality for gray and color spaces

def facial_Feature(image, gray, x, y, w, h):

    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
    eye_y = []
    
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2) # Drawing eye detection label rectangles
        eye_y.append(ey + (eh/2) )

    return roi_color, roi_gray, eye_y

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

# finds straight lines from the image 
def line_Getter(img, gray, eye_avg):
    blurred_gray = cv2.GaussianBlur(gray, (5,5),0) # add a blur to ignore background of some image
    edges = cv2.Canny(blurred_gray, 26, 115) # apply canny edge detection on image
    cv2.imshow("Edged Image", edges) # Drawing canny edge lines 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=75, maxLineGap=20) # detects all straight lines from the canny edges (returns array of lines)
    final_Lines = []
    if len(lines) != 0: 
        for line in lines: 
            x1, y1, x2, y2 = line[0]
            if(y1 >= eye_avg and y2 >= eye_avg): # only show lines under eyes
                cv2.line(img, (x1,y1), (x2, y2), (255,0,0),3) # Drawing all houglines
                final_Lines.append(line[0])
    return final_Lines


# Detects the top, left, and right edges of the mask
def mask_Creator(lines,img):
    y_Max = 99999
    x_Max = -1 # actually our min but not touching 
    x_Min= 9999999 # actually the max X value
    for line in lines:
        x1, y1, x2, y2 = line
        if(max(x1, x2) >= x_Max):
            x_Max = max(x1, x2)
        if(min(x1, x2) <= x_Min):
            x_Min = min(x1, x2)
        
        if(max(y1, y2) <= y_Max):
            y_Max = max(y1,y2)

    #cv2.line(img, (x_Min,y_Max), (x_Max, y_Max), (0,0,255),3) # Drawing main houghline

    mask_img = np.zeros(img.shape, dtype="uint8")
    w, h, c  = img.shape
    print(img.shape)
    print(mask_img.shape)
    print(x_Max)
    print(x_Min)
    print(y_Max)
    cv2.rectangle(mask_img, (x_Min, y_Max), (x_Max,h), (0,0,255), -1) # Drawing mask rectangle
    
    cv2.waitKey(0)
    
    return mask_img
    

# MAIN FUNCTION, LORD FORGIVE ME FOR WHAT I'M ABOUT TO CODE
images = load_images_from_folder("images")
labels_dict={0:'without mask',1:'mask'}
color_dict={0:(0,0,255),1:(0,255,0)}
size = 4
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for image in images:
    imagePath = image
    cv2.imshow("Input",imagePath)
    mini = cv2.resize(image, (image.shape[1] // size, image.shape[0] // size)) # make a smaller image

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
            cv2.imshow("face", image)
            cropped_image, cropped_gray, eye_y = facial_Feature(image, gray, x, y, w, h)

            eye_avg = (sum(eye_y)/len(eye_y))+ 15 # get average of eyes and look just below
            final_Lines= line_Getter(cropped_image, cropped_gray, eye_avg)
            mask_img=mask_Creator(final_Lines, cropped_image)
            
            cv2.imshow("Image", cropped_image)
            cv2.imshow("Mask", mask_img)
            # res = cv2.bitwise_and(cropped_image,cropped_image,mask = mask_img)
            dst = cv2.addWeighted(cropped_image,0.5,mask_img,1,0)
            cv2.imshow("Combined", dst)
            # NEW  CODE FOR OUTPUT GENERATION
            assert cropped_image.shape == mask_img.shape

            h, w, _ = cropped_image.shape
            grid = 8
            cropped_image = cropped_image[:h//grid*grid, :w//grid*grid, :]
            mask_img = mask_img[:h//grid*grid, :w//grid*grid, :]
            print('Shape of cropped_image: {}'.format(cropped_image.shape))

            cropped_image = np.expand_dims(cropped_image, 0)
            mask_img = np.expand_dims(mask_img, 0)
            input_image = np.concatenate([cropped_image, mask_img], axis=2)
            FLAGS = ng.Config('inpaint.yml')
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            with tf.Session(config=sess_config) as sess:
                input_image = tf.constant(input_image, dtype=tf.float32)
                output = model.build_server_graph(FLAGS, input_image)
                output = (output + 1.) * 127.5
                output = tf.reverse(output, [-1])
                output = tf.saturate_cast(output, tf.uint8)
                # load pretrained model
                vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                assign_ops = []
                for var in vars_list:
                    vname = var.name
                    from_name = vname
                    var_value = tf.contrib.framework.load_variable('CC:/Users/zacca/OneDrive/Desktop/Year 4/Capstone/Code/CapstoneFaceMaskRemoval/model_logs/release_celeba_hq_256_deepfill_v2', from_name)
                    assign_ops.append(tf.assign(var, var_value))
                sess.run(assign_ops)
                print('Model loaded.')
                result = sess.run(output)
                cv2.imshow("Output", result[0][:, :, ::-1])
            cv2.waitKey(0)
        #else:
            
            # Mask not detected
            # ctypes.windll.user32.MessageBoxW(0, "Mask not Detected", "No Masks", 1)
            cv2.waitKey(0)
        #else:
            
            # Mask not detected
            # ctypes.windll.user32.MessageBoxW(0, "Mask not Detected", "No Masks", 1)