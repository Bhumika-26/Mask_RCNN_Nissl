# Jan 23, 2023
# Samik Banerjee
# This unit-test code is for Overlaying two images (Grayscale/Binary over RGB)
# python3 overlay_2images.py <RGB Image Path> <Single-channel image Path> <Transperancy b/w 0,1> <Output Folder>


from PIL import Image
from skimage.io import imread, imsave
import os
import numpy as np
import sys
import cv2

# Code for Reading JP2 images using KAKADU, needs a temp folder in CWD
def imread_fast(img_path):
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path) 
    base = base[0:-4]
    err_code = os.system("kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 16") # Needs a temp folder for intermediate TIFF image in the CWD
    img = imread('temp/'+base+'.tif')
    os.system("rm temp/"+base_C+'.tif')
    return img

# Code for Writinging JP2 images using KAKADU, needs a temp folder in CWD
def imwrite_fast(img_path, opImg):
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path)
    base = base[0:-4]
    img = imsave('temp/'+base+'.tif', opImg) # Needs a temp folder for intermediate TIFF image in the CWD
    err_code = os.system("kdu_compress -i temp/"+base_C+".tif -o "+img_path_C+" -rate 1 Creversible=yes Clevels=7 Clayers=8 Stiles=\{1024,1024\} Corder=RPCL Cuse_sop=yes ORGgen_plt=yes ORGtparts=R Cblk=\{32,32\} -num_threads 32")
    os.system("rm temp/"+base_C+'.tif')

imagePath = sys.argv[1] # Input Image Path
maskPath = sys.argv[2] # Input Mask Path
alpha = float(sys.argv[3]) # Transperancy for overlay
opPath = sys.argv[4] # Output Path
fileExt = sys.argv[5]

print("Reading Image")
if fileExt=='jp2':
    img = imread_fast(imagePath)
else:
    img = imread(imagePath) 
w, h, c = img.shape

print("Reading Mask")
if fileExt=='jp2':
    maskRead = imread_fast(maskPath)
else:
    maskRead = imread(maskPath) 
maskShp = maskRead.shape
if len(maskShp)==2:
    mask = np.zeros((maskShp[0], maskShp[1], 3), dtype = 'uint8')  
    mask[:,:,0] = maskRead
else:
    mask = maskRead

print("Resizing Mask")
mask = cv2.resize(mask,(h,w), interpolation=cv2.INTER_LANCZOS4)

print("Overlaying Images")
op1 = np.uint8(mask * alpha)
del mask
print("Mask Transperancy Created") 
op2 = np.uint8(img * (1 - alpha)) 
del img
print("Image Transperancy Created")
op = op1 + op2
del op1, op2

print("Writing Overlayed image")
if fileExt=='jp2':
    imwrite_fast(os.path.join(opPath,"mask.jp2"), op)
else:
    imsave(os.path.join(opPath,"mask.tif"), op)