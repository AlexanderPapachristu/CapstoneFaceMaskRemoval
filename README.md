# Capstone FaceMaskRemoval

The global pandemic known as COIVD-19 has been part of our lives for over the past two years,
and alongside COVID-19 came the global-wide mask mandate policy which continues to be
enforced in some parts of the world. As it was widely noticed, computer vision – such as the
kind used on phones to biometrically authorise a user – had a hard time reliably recognizing a
face with a face mask on. 

This Project will allow a user to insert an input image of a subject wearing a face mask and output an image with and AI generated face where the face mask used to be.

### Before Running
For Windows 10 <br/>
Download Anaconda <br />
https://www.anaconda.com/products/individual#windows

### Next, 

Create an environment in anaconda with python 3.7 using the following code and enter the environment
<br /> `conda create -n FaceMaskRemoval python==3.7` <br />
`conda activate FaceMaskRemoval` 

Once inside the environment run 
`pip install -r requirements.txt`

Once the requirements are installed, download the pretrained model in the following link
fine-tuned weights: [CelebA-HQ](https://drive.google.com/u/0/uc?id=17oJ1dJ9O3hkl2pnl8l2PtNVf2WhSDtB7&export=download) (for `networks.py`) (This file will go into the pretrained folder)

Once the download is complete put those folders in the CapstoneFaceMaskRemoval folder

Then run initial.py with `python initial.py`

Click on the upload file and choose the picture you would like to remove the facemask from and hit Recreate image. A sample of 10 images will appear and you will be left to choose the output you think is best. If you wish to select another file to upload, first hit the clear button and then upload another file.  

Here are some sample images 
![image](https://user-images.githubusercontent.com/55794234/229315856-2bf397ad-6af6-4075-af23-802ac7afb3dc.png)
![image](https://user-images.githubusercontent.com/55794234/229315859-b2f962e4-999b-454e-b734-bfb86f98e896.png)
![image](https://user-images.githubusercontent.com/55794234/229315863-29794a48-b0a5-4064-b873-6e9ad255eb0f.png)





