# Jan 18, 2023
# Samik Banerjee
# This unit-test code is for Nissl Cell-Detection BrightFeild -- Runs on GPU ID 1 
# Create a folder with the images for detection and pass as parameter 'inputLocationFolder'
# JSON Outputs will be in 'M32/Cell_Detection/CellDetPass1_Nissl/<brainName>/'
# Mask Outputs will be in 'M32/Cell_Detection/CellDetPass1_Nissl/Annotation_tool/<brainName>/'
# Intermediate debug outputs will be in  '$CWD/temp_out/' & '$CWD/temp_mask/'
# 'fileExtension' parameter should be without '.' 
# python3 ajay_nissl_dir_1.py <brainName> <inputLocationFolder> <fileExtension> <lossyExists(0/1)>


# In[1]:
# Important Imports:
import ajay_nucleus0 as nucleus1
from mrcnn.config import Config
from mrcnn import model as modellib
import os
import sys
import numpy as np
import cv2
import numpy as np
import os
import sys
from skimage.io import imread, imsave
import h5py
import hdf5storage
from scipy.ndimage.morphology import binary_fill_holes
from timer import Timer
import fnmatch

# Imports that may not useful but used in development process. #DELETE if you want in Production Code
from skimage.measure import find_contours
from matplotlib.pyplot import imshow
import bisect
import statistics
from scipy import misc 
from scipy.io import loadmat
import pickle
import matplotlib.path as mplPath
from multiprocessing import Pool
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import active_contour
from skimage.morphology import skeletonize
from skimage.morphology import convex_hull_image
from functools import partial
from time import gmtime, strftime
import matplotlib.image as mpimg
import glob
import matplotlib.pyplot as plt
import re
import skimage.io
import tensorflow as tf
import keras.backend as K
from PIL import Image
import json
import datetime
from mrcnn import visualize
from mrcnn import utils


# In[2]:
def imread_fast(img_path):
    img_path_C= img_path.replace("&", "\&")
    base_C = os.path.basename(img_path_C)
    base_C = base_C[0:-4]
    base = os.path.basename(img_path)
    base = base[0:-4]
    err_code = os.system("kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 16")
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

# In[4]:
class NucleusConfig(Config):
    """Configuration for training on the nissl segmentation dataset. while using command "train"
    """
    # Give the configuration a recognizable name
    NAME = "nissl"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1 # since multiprocessing doesn't work

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + cell

    # Number of training and validation steps per epoch
    # 238 is the size of train set
    STEPS_PER_EPOCH = 218 // (IMAGES_PER_GPU * GPU_COUNT)
    VALIDATION_STEPS = max(1, 20 // (IMAGES_PER_GPU * GPU_COUNT))

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nissl and BG
    DETECTION_MIN_CONFIDENCE = 0.7

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2, 4]

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    # setting means for R and G to 0 since cells are mostly blue
    MEAN_PIXEL = np.array([0, 0, 103.9])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    IMAGE_CHANNEL_COUNT = 3

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 400

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 800
# In[5]:
class NucleusInferenceConfig(NucleusConfig):
    """Test-time configurations. while using command "test" or "detect"
    """
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9
    USE_MINI_MASK = False

# In[6]:
def image2mask(image, mask, w, h, fN):
    sz = 512
    op = np.zeros((w,h),dtype='bool')
    for row in range(0, w, sz):
        for col in range(0, h, sz):
            tile = image[row:min(row + sz, w), col:min(col + sz, h), :]
            tileM = mask[row:min(row + sz, w), col:min(col + sz, h)]
            wT, hT, cT = tile.shape
            if np.sum(tileM):
                ipTile = np.zeros((512, 512, 3), dtype= 'uint8')
                ipTile[0:wT, 0:hT, :] = tile 
                out = nucleus1.detect(model, ipTile)
                # out = outR["masks"]
                # rois = outR["rois"]
                if(out.shape[2]):
                    for idx in range(1,out.shape[2]+1):
                        cell = np.uint8(out[:,:,idx-1])
                        _, contours, _ = cv2.findContours(cell,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        if contours:
                            sN = "\n["
                            for verts in contours[0]:
                                sN = sN + "\n[" + str(int(verts[0][1])+row) + "," + str((int(verts[0][0])+col)*-1) + "],"
                            fN.write(sN[:-1] + "\n],")
                    outK = np.zeros((sz, sz), dtype= 'float')
                    outK[np.sum(out,axis=2).nonzero()] = 1
                    outK = cv2.threshold(np.uint8(outK), 0, 1, cv2.THRESH_BINARY)
                    outK = np.asarray(outK[1])
                    outK = binary_fill_holes(outK)
                    op[row:min(row + sz, w), col:min(col + sz, h)] = outK[0:wT, 0:hT]  
    # t.stop()
    return op


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

# In[7]:
weights_path = '/home/samik/Mask_RCNN/samples/nucleus/mask_rcnn_nucleus_NISSL.h5'
# weights_path = '/home/samik/Mask_RCNN/logs/nucleus20191030T1510/mask_rcnn_nucleus_0199.h5'
os.environ['OPENCV_IO_ENABLE_JASPER']= '1'
config = NucleusInferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir='/home/samik/Mask_RCNN/logs/')

# In[9]:
model.load_weights(weights_path, by_name=True)
t = Timer()

# In[11]:
brainNo = sys.argv[1]

# In[12]:
filePath = sys.argv[2]
fileExt = sys.argv[3]
fileLossy = int(sys.argv[4])
# In[13]:
outDir = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_Nissl/'+ brainNo + '/'
jsonOutN = os.path.join(outDir, 'jsonN/')
maskOut = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_Nissl/Annotation_tool/' + brainNo + '/'

# In[14]:
os.system("mkdir " + outDir)
os.system("mkdir " + maskOut)
os.system("mkdir " + jsonOutN)

# In[15]:
argLossy = ""
if fileLossy:
    argLossy = "_lossy"
fileList1 = os.listdir(filePath)
# print(fileList1)
for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*' + argLossy + '.' + fileExt)):
        fileList1.remove(fichier) 
# print(fileList1)
for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*N*')):
        fileList1.remove(fichier)
# print(fileList1)
# In[16]:
fileList2 = [] #os.listdir(maskOut)

# In[73]:
for files in fileList1:
    
    if files not in fileList2:
        print(files)
        if fileExt=='jp2':
            image = imread_fast(os.path.join(filePath, files))
        else:
            image = imread(os.path.join(filePath, files)) 
        w, h, c = image.shape
        # mask = np.ones((w,h) , dtype = 'uint8')
        # # maskB = hdf5storage.loadmat(os.path.join(maskDir,files.replace('jp2', 'mat')))['seg']
        # maskB = maskB / maskB.max()
        image = image.astype(np.uint8)

        mask = pre_proc(image, w,h)
        
        # Debug point to see mask... #COMMENT on Running in Production mode
        # if fileExt=='jp2':
        #     imwrite_fast(os.path.join('temp_mask', files), mask)
        # else:
        #     imsave(os.path.join('temp_mask', files), mask) 
        # print("Mask Written!!!!!!!!!!")

        # Debug point to see masked image... #COMMENT on Running in Production mode
        # if fileExt=='jp2':
        #     imwrite_fast(os.path.join('temp_out', files), image) 
        # else:
        #     imsave(os.path.join('temp_out', files), image)     
        # print("Masked Image Written!!!!!!!!!!")

        # Write out the Geojson Corrdinates for each cell in the Nissl
        fN = open(os.path.join(jsonOutN, files.replace(fileExt, 'json')), "w")
        fN.write("{\"type\":\"FeatureCollection\",\"features\":\n[{\"type\":\"Feature\",\n\"id\":\"0\",\n\"properties\":\n{\"name\":\"Nissl Cell\"},\n\"geometry\":\n{\"type\":\"MultiPolygon\",\n\"coordinates\":\n[")
        
        
        op = image2mask(image, mask,  w, h, fN)
        del mask
        del image

        op = np.uint8(op) * 255


        fN.write("[]\n]\n}\n}\n]\n}")
        fN.close()

         
        if fileExt=='jp2': 
            imwrite_fast(os.path.join(maskOut, files), op)
        else:
            imsave(os.path.join(maskOut, files), op) 
        del op
