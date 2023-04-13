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
from skimage.io import imread
from scipy import misc 
from scipy.io import loadmat
import h5py
import hdf5storage
import bisect
import statistics
from scipy.ndimage.morphology import binary_fill_holes




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


def image2mask(image, maskB, op,  w, h):
    sz = 512
    for row in range(0, w - sz - 1, sz):
        # print(row)
        for col in range(0, h - sz - 1, sz):
            tile = image[row:row + sz, col:col + sz, :]
            tileM = maskB[row:row + sz, col:col + sz]
            if np.sum(tileM): 
                tile = cv2.resize(tile, (512,512))
                tileN = tile
                out = nucleus1.detect(model, tileN)

                outK = np.zeros((512, 512), dtype= 'float')
                outK[np.sum(out,axis=2).nonzero()] = 1
                outK = cv2.resize(np.uint8(outK), (sz,sz))
                outK = cv2.threshold(outK, 0, 1, cv2.THRESH_BINARY)
                outK = np.asarray(outK[1])
                outK = binary_fill_holes(outK)

                op[row:row + sz, col:col + sz] = outK               
    return op

weights_path = '/home/samik/Mask_RCNN/samples/nucleus/mask_rcnn_nucleus_0110.h5'
# weights_path = '/home/samik/Mask_RCNN/logs/nucleus20191030T1510/mask_rcnn_nucleus_0199.h5'
os.environ['OPENCV_IO_ENABLE_JASPER']= '1'
config = NucleusInferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir='/home/samik/Mask_RCNN/logs/')

model.load_weights(weights_path, by_name=True)

brainNo = 'MD806'

# filePath = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/' + brainNo + '_img/' #reg_high_tif_pad_jp2/'
# maskDir = '/nfs/data/main/M32/RegistrationData/Data/' + brainNo + '/Transformation_OUTPUT/reg_high_seg_pad/'
#filePath = '/nfs/data/main/M25/marmosetRIKEN/NZ/m6328/m6328F/JP2/'
filePath = '/nfs/data/main/M28/mba_converted_imaging_data/' + brainNo + '/lossyNOTdown/'

outDir = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_Nissl/'+ brainNo + '/'
# outDir = '/home/samik/Mask_RCNN/samples/nucleus/temp_out/'
# jsonOutR = os.path.join(outDirR, 'jsonR/')
maskOut = '/nfs/data/main/M32/Cell_Detection/CellDetPass1_Nissl/Annotation_tool/' + brainNo + '/'
# maskOutC = os.path.join(outDir, 'maskC/')


os.system("mkdir " + outDir)
os.system("mkdir " + maskOut)
# os.system("mkdir " + jsonOutR)
# os.system("mkdir " + jsonOutG)

fileList1 = os.listdir(filePath)
fileList2 = [] #os.listdir(maskOut)

for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*.jp2')):
        fileList1.remove(fichier)

for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
    if not(fnmatch.fnmatch(fichier, '*N*')):
        fileList1.remove(fichier)

for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
        if fichier in fileList2[:]:
            fileList1.remove(fichier)

#print(fileList1)

for files in fileList1:
    print(files)
    image = imread_fast(os.path.join(filePath, files))
    w, h, c = image.shape
    maskB = np.ones((w,h) , dtype = 'uint8')
    # maskB = hdf5storage.loadmat(os.path.join(maskDir,files.replace('jp2', 'mat')))['seg']
    maskB = maskB / maskB.max()

    image = image.astype(np.uint8)
    _,maskB = cv2.threshold(maskB,0,1,cv2.THRESH_BINARY)
           
    # w, h, c = image.shape
    # maskBG = np.ones((w,h) , dtype = 'uint8')
    maskB = np.uint8(maskB) * 255

    op = np.zeros((w,h),dtype='bool')
    op = image2mask(image, maskB, op, w, h)
    op = np.uint8(op) * 255
    opC = np.zeros((w,h,c),dtype='uint8')
    opC[:,:,2] = op

    #_, thresh = cv2.threshold(opG,127,255,cv2.THRESH_BINARY)
    #_, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(thresh))
    #f = open(os.path.join(jsonOutG, files.replace('jp2', 'json')), "w")
    #f.write("{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\",\"id\":\"0\",\"properties\":{\"name\":\"Green Cell\"},\"geometry\":{\"type\":\"MultiPoint\",\"coordinates\":[")
    #for pts in centroids:
    #    pts64 = pts.astype(np.int64)
    #    f.write("[" + str(pts64[0]) + "," + str(pts64[1]*-1) + "],")
    #f.write("[]]}}]}")
    #f.close()

    cv2.imwrite(os.path.join(maskOut, files.replace('_lossy.jp2', '.jp2')), op)
    cv2.imwrite(os.path.join(outDir, files.replace('_lossy.jp2', '.jp2')), opC)








