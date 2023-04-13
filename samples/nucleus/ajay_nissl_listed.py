#!/usr/bin/env python
# coding: utf-8

# In[1]:
import ajay_nucleus0 as nucleus1
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import fnmatch
import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import tensorflow as tf
import keras.backend as K
# from imgaug import augmenters as iaa
from PIL import Image
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import re
import os
from skimage.measure import find_contours
from skimage.io import imread
from multiprocessing import Pool
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import active_contour
from skimage.morphology import skeletonize
from skimage.morphology import convex_hull_image
from functools import partial
from time import gmtime, strftime
import matplotlib.image as mpimg
import pickle
import matplotlib.path as mplPath
import sys
from skimage.io import imread, imsave
from scipy import misc 
from scipy.io import loadmat
import h5py
import hdf5storage
import bisect
import statistics
from scipy.ndimage.morphology import binary_fill_holes
from timer import Timer
from matplotlib.pyplot import imshow

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
def image2mask(image, w, h, fN):
    sz = 512
    op = np.zeros((w,h),dtype='bool')
    # t.start()
    for row in range(0, w - sz - 1, sz):
        # print(str(row) +"/" +str(w))
        for col in range(0, h - sz - 1, sz):
            tile = image[row:row + sz, col:col + sz, :]
            if np.sum(tile[:, :, 2]): 
                out = nucleus1.detect(model, tile)
                if(out.shape[2]):
                    for idx in range(1,out.shape[2]+1):
                        cell = np.uint8(out[:,:,idx-1])
                        _, contours, _ = cv2.findContours(cell,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        if contours:
                            sN = "\n["
                            for verts in contours[0]:
                                sN = sN + "\n[" + str(int(verts[0][1])+col) + "," + str((int(verts[0][0])+row)*-1) + "],"
                            fN.write(sN[:-1] + "\n],")
                    outK = np.zeros((512, 512), dtype= 'float')
                    outK[np.sum(out,axis=2).nonzero()] = 1
                    outK = cv2.threshold(np.uint8(outK), 0, 1, cv2.THRESH_BINARY)
                    outK = np.asarray(outK[1])
                    outK = binary_fill_holes(outK)
                    op[row:row + sz, col:col + sz] = outK   
    # t.stop()
    return op


def pre_proc(image, w,h):
    mask1 = 255 - image[:, :, 0]
    mask2 = 255 - image[:, :, 1]
    _, mask1 = cv2.threshold(mask1, 32, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(mask2, 32, 255, cv2.THRESH_BINARY)
    mask = np.uint8(mask1 + mask2)
    mask = cv2.resize(mask, (int(h / 100), int(w / 100)))
    mask = cv2.medianBlur(mask, 11)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)), iterations = 1)
    mask = cv2.GaussianBlur(mask, (5,5), 0.5)
    nw,nh = mask.shape
    mask[0:int(0.1*nw), :] = 0
    mask[:,0:int(0.05*nh)] = 0
    mask[nw-int(0.1*nw) : nw, :] = 0
    mask[:, nh-int(0.05*nh) : nh] = 0
    _, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)

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
listmin = int(sys.argv[2])
listmax = int(sys.argv[3])

# In[11]:
brainNo = sys.argv[1]

# In[12]:
filePath = '/nfs/data/main/M28/mba_converted_imaging_data/' + brainNo + '/lossyNOTdown/'

# In[13]:
outDir = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_Nissl/'+ brainNo + '/'
jsonOutN = os.path.join(outDir, 'jsonN/')
maskOut = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_Nissl/Annotation_tool/' + brainNo + '/'

# In[14]:
os.system("mkdir " + outDir)
os.system("mkdir " + maskOut)
os.system("mkdir " + jsonOutN)

# In[15]:
f=open('/nfs/data/main/M32/Samik/listMD806.txt')
fileList1=f.readlines()[listmin: listmax]
print(fileList1)

# In[16]:
fileList2 = [] #os.listdir(outDir)

# In[73]:
for files in fileList1:
    files = os.path.split(files)[1].replace('\n', '')
    if files not in fileList2:
        print(files)
        image = imread_fast(os.path.join(filePath, files))
        w, h, c = image.shape
        # maskB = np.ones((w,h) , dtype = 'uint8')
        # # maskB = hdf5storage.loadmat(os.path.join(maskDir,files.replace('jp2', 'mat')))['seg']
        # maskB = maskB / maskB.max()
        image = image.astype(np.uint8)

        mask = pre_proc(image, w,h)


        image[:,:,0] = np.multiply(image[:,:,0], mask) # Mask out regions outside the brain tissue in the Red Channel 
        image[:,:,1] = np.multiply(image[:,:,1], mask) # Mask out regions outside the brain tissue in the Green Channel 
        image[:,:,2] = np.multiply(image[:,:,2], mask) # Mask out regions outside the brain tissue in the Blue Channel 
        del mask

        # Write out the Geojson Corrdinates for each cell in the Nissl
        fN = open(os.path.join(jsonOutN, files.replace('_lossy.jp2', '.json')), "w")
        fN.write("{\"type\":\"FeatureCollection\",\"features\":\n[{\"type\":\"Feature\",\n\"id\":\"0\",\n\"properties\":\n{\"name\":\"Nissl Cell\"},\n\"geometry\":\n{\"type\":\"MultiPolygon\",\n\"coordinates\":\n[")
        
        
        op = image2mask(image, w, h, fN)
        del image

        op = np.uint8(op) * 255


        fN.write("[]\n]\n}\n}\n]\n}")
        fN.close()

        imwrite_fast(os.path.join(maskOut, files.replace('_lossy.jp2', '.jp2')), op)
        del op
