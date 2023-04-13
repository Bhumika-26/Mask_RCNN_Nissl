# Jan 26, 2023
# Samik Banerjee
# This code is for tiling large images
# Needs a temp folder for intermediate TIFF image in the CWD
# Create a folder with the images for detection and pass as parameter 'inputLocationFolder'
# Ouput Tiles <inputLocationFolder>/image_tiles/<fileName w/o fileExtension >/*.<fileExtension>
# python3 tiling_large_images.py <brainName> <inputLocationFolder> <fileExtension> 


# In[1]:
# Important Imports:
from math import ceil, floor
import os
import sys
import numpy as np
import os
import sys
from skimage.io import imread, imsave
import cv2


# In[2]:
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

# In[3]:
def imwrite_fast(img_path, opImg):
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path)
    base = base[0:-4]
    img = imsave('temp/'+base+'.tif', opImg) # Needs a temp folder for intermediate TIFF image in the CWD
    err_code = os.system("kdu_compress -i temp/"+base_C+".tif -o "+img_path_C+" -rate 1 Creversible=yes Clevels=7 Clayers=8 Stiles=\{1024,1024\} Corder=RPCL Cuse_sop=yes ORGgen_plt=yes ORGtparts=R Cblk=\{32,32\} -num_threads 32")
    os.system("rm temp/"+base_C+'.tif')

def pre_proc(image, w,h):
    mask1 = 255 - image[:, :, 0]
    mask2 = 255 - image[:, :, 1]
    _, mask1 = cv2.threshold(mask1, 16, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(mask2, 16, 255, cv2.THRESH_BINARY)
    mask = np.uint8(mask1 + mask2)
    mask = cv2.resize(mask, (int(h / 100), int(w / 100)))
    mask = cv2.medianBlur(mask, 11)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)), iterations = 1)
    mask = cv2.GaussianBlur(mask, (5,5), 0.5)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    mask = cv2.resize(mask,(h,w))
    return mask

# In[11]:
brainNo = sys.argv[1] # dummy for future use

# In[12]:
filePath = sys.argv[2] # Input Folder Location
fileExt = sys.argv[3]
# In[13]:
outDir = os.path.join(filePath + 'image_tiles/')

# In[14]:
os.system("mkdir " + outDir)

# In[15]:
# Remove other files without fileExt if present
fileList1 = os.listdir(filePath)
for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fichier.endswith(fileExt)):
        fileList1.remove(fichier) 

# In[16]:
fileList2 = [] #os.listdir(outDir) # TODO: check for future use

# In[73]:
for files in fileList1:
    
    if files not in fileList2:
        print(files)
        sz = 4096 # Change if required default 4096 X 4096 tiles (except the extremties)
        if fileExt=='jp2':
            image = imread_fast(os.path.join(filePath, files))
        else:
            image = imread(os.path.join(filePath, files)) 
        w, h, c = image.shape

        tiledir = os.path.join(outDir,files.replace('.' + fileExt, ''))
        maskdir = os.path.join(tiledir, 'mask') 
        os.system("mkdir " +  tiledir)
        os.system("mkdir " +  maskdir)
        mask = pre_proc(image, w,h)

        for row in range(0, w, sz):
            for col in range(0, h, sz):
                tile = image[row:min(row + sz, w), col:min(col + sz, h), :]
                maskT = mask[row:min(row + sz, w), col:min(col + sz, h)]
                tileName = str(floor(row/sz)+1) + "_" + str(floor(col/sz)+1) + "." + fileExt # fileName w/o fileExtension 
                print(tileName)
                if fileExt=='jp2': 
                    imwrite_fast(os.path.join(tiledir, tileName), tile)
                else:
                    imsave(os.path.join(tiledir, tileName), tile) 
                imsave(os.path.join(maskdir, tileName.replace(fileExt, 'tif')), maskT)
