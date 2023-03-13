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
import tensorflow as tf
import torchvision.transforms as T
import torch

import tkinter as tk
import tk_tools
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

#Uses pre-defined haar cascades to detect facial features
#detectMultiScale detects different object sizes and labels them
#Region of interest functionality for gray and color spaces

def facial_Feature(image, gray):

    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    roi_gray = gray
    roi_color = image
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
    eye_y = []

    #***********************************#
    # Need for eyeglasses detection but #
    #      currently not working        #
    #***********************************#

    # if len(eyes) == 0:
    #     eyes_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    #     eyes = eyes_cascade.detectMultiScale(roi_gray)


    for (ex,ey,ew,eh) in eyes:
            eye_y.append(ey + (eh/2) )

       
    return roi_color, roi_gray, eye_y

#imports images and resizes them to a standard size

def load_images_from_folder(path):
    
    images = []
    
    for item in os.listdir(path):
        img = cv2.imread(os.path.join(path,item))
       
        if img is not None:
            images.append(img)

    return images

# finds straight lines from the image 
def line_Getter(img, gray, eye_avg):
    blurred_gray = cv2.GaussianBlur(gray, (5,5),0) # add a blur to ignore background of some image
    edges = cv2.Canny(blurred_gray, 26, 115) # apply canny edge detection on image
    # cv2.imshow("Edged Image", edges) # Drawing canny edge lines 

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 32, minLineLength=9, maxLineGap=2) # detects all straight lines from the canny edges (returns array of lines)

    w = img.shape[1]
    final_Lines = []
    if len(lines) != 0: 
        for line in lines: 
            x1, y1, x2, y2 = line[0]
            if(y1 >= eye_avg and y2 >= eye_avg  and x1 > 0 + 50 and x2 > 0 + 50  and x1 < w - 50 and x2 < w - 50 ): # only show lines under eyes
                final_Lines.append(line[0])
    return final_Lines


# Detects the top, left, and right edges of the mask
def mask_Creator(lines,img):
    y_Max =-1
    y_Min = 99999
    x_Max = -1 # actually our min but not touching 
    x_Min= 9999999 # actually the max X value
    for line in lines:
        x1, y1, x2, y2 = line
        if x2-x1 != 0:
            slope =  abs(y2-y1) / abs(x2-x1)
        else:
            slope = 2
        if(slope < 1 and  max(y1, y2) >= y_Max):
            y_Max = max(y1,y2)
        
        if( min(y2,y1)<=y_Min):
            y_Min = min(y2,y1)

    print(f"y_min: {y_Min}  y_max: {y_Max}")
    for line in lines:
        x1, y1, x2, y2 = line
        y_Max_temp = y_Max
        if x2-x1 != 0:
            slope =  abs(y2-y1) / abs(x2-x1)
        else:
            slope = 0
        # print(slope)
        if(max(x1, x2) >= x_Max and (y2 < y_Max_temp and y1 < y_Max_temp ) and slope > 1):
            x_Max = max(x1, x2)
        if(min(x1, x2) <= x_Min and (y2 < y_Max_temp and y1 < y_Max_temp ) and slope > 1):
            x_Min = min(x1, x2)

    mask_img = np.zeros(img.shape, dtype="uint8")
    h = img.shape[0]
    cv2.rectangle(mask_img, (x_Min, y_Max  ), (x_Max,y_Min-19 ), (255,255,255), -1) # Drawing mask rectangle
    
    return mask_img

def output_Creator(cropped_image, mask_img):
    # # set up network
   
    generator_state_dict = torch.load("pretrained/states_pt_celebahq.pth", map_location= torch.device('cpu'))['G']


    from model.networks import Generator
 

    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available()
                          and use_cuda_if_available else 'cpu')
    # # # set up network
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)

    generator_state_dict = torch.load("pretrained/states_pt_celebahq.pth", map_location= torch.device('cpu'))['G']
    generator.load_state_dict(generator_state_dict, strict=True)
    img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(mask)
    # load image and mask
    image = im_pil
    mask = mask_pil

    # prepare input
    image = T.ToTensor()(image)
    mask = T.ToTensor()(mask)

    _, h, w = image.shape
    grid = 8

    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    print(f"Shape of image: {image.shape}")

    image = (image*2 - 1.).to(device)  # map image values to [-1, 1] range
    mask = (mask > 0.5).to(dtype=torch.float32,
                           device=device)  # 1.: masked 0.: unmasked

    image_masked = image * (1.-mask)  # mask image

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x*mask],
                  dim=1)  # concatenate channels

    with torch.inference_mode():
        _, x_stage2 = generator(x, mask)

    # complete image
    image_inpainted = image * (1.-mask) + x_stage2 * mask
    # save inpainted image
    img_out = ((image_inpainted[0].permute(1, 2, 0) + 1)*127.5)
    img_out = img_out.to(device='cpu', dtype=torch.uint8)
    img_out = Image.fromarray(img_out.numpy())
    img_out.show()

def reject_outliers(data, m=6.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m].tolist()

# MAIN FUNCTION, LORD FORGIVE ME FOR WHAT I'M ABOUT TO CODE

window = tk.Tk()
window.geometry("640x360")

image_prompt = tk.Label(text = "Please select an image")
image_prompt.grid(row=1,column=1)
image_prompt.pack

b1 = tk.Button(window, text='Upload File', width=20,command = lambda:upload_file())
b1.grid(row=2,column=1)
globalimage=[]

def upload_file():
    global tkImage
    filename = filedialog.askopenfilename(filetypes=[("Image file", ".jpg .png")])
    tkImage = ImageTk.PhotoImage(file=filename)

    img = cv2.imread(filename)
    globalimage.append(img)

    b2 = tk.Button(window, image=tkImage)
    b2.grid(row=3,column=1)

window.mainloop()

images = globalimage
labels_dict={0:'without mask',1:'mask'}
color_dict={0:(0,0,255),1:(0,255,0)}
size = 4
#classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = images[0]
print(image.shape)
# cv2.imshow("Input",imagePath)
inter = cv2.INTER_AREA
height = 256
dim = None
(h, w) = image.shape[:2]
r = height / float(h)
dim = (int(w * r), height)
image = cv2.resize(image, dim, interpolation = inter)

# image = cv2.resize(image, (512,512), interpolation = inter)

# cv2.imwrite("OUTPUT/image.png",image)
# image = cv2.resize(image, (image.shape[1] // size, image.shape[0] // size)) # make a smaller image
print(image.shape)
# detect MultiScale / faces
# faces = classifier.detectMultiScale(mini)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Input", image)
cropped_image, cropped_gray, eye_y = facial_Feature(image, gray)
if len(eye_y) > 0:
    
    # Remove outliner eyes
    eye_y = np.array(eye_y)
    eye_y = reject_outliers(eye_y)
    print(f"EYE Y: {eye_y}")
    eye_avg = (sum(eye_y)/len(eye_y)) + 22 # get average of eyes and look just below
    
else:
    print("NO EYES")
    eye_avg = int( cropped_image.shape[0]/2) + 10
final_Lines= line_Getter(cropped_image, cropped_gray, eye_avg)
cv2.imshow("TESR", cropped_image)
mask_img=mask_Creator(final_Lines, cropped_image)
# cv2.imshow("Image", cropped_image)
cv2.imshow("Mask", mask_img)
# cv2.imwrite("OUTPUT/mask.png", mask_img)
# res = cv2.bitwise_or(cropped_image,cropped_image,mask = mask_img)
dst = cv2.addWeighted(cropped_image,1,mask_img,1,0)
cv2.imshow("Combined", dst)
# # NEW  CODE FOR OUTPUT GENERATION
if mask_img is not None:
    output_Creator(cropped_image, mask_img)
else:
    print("NO MASK")
cv2.waitKey(0)
