# +
# Samik Banerjee
# Dated : 01 Sep 2021
# fileJson = sys.argv[1] 
# width = sys.argv[2] 
# height = int(sys.argv[3])
# pixel_resolution = int(sys.argv[4])
# -


import numpy as np
from skimage.util import dtype
import os
import cv2
import sys
from skimage.io import imread, imsave
import time
import json
import numpy as np


# file = sys.argv[1]
# os.environ['OPENCV_IO_ENABLE_JASPER']= '1'
# szX = int(sys.argv[2])
# szY = int(sys.argv[3])
# resolution = float(sys.argv[4])
# file = '/home/samik/ProcessDet/morse_code/tempIP/atlas-seg_to_M1229-F84--_1_0150.geojson'
# szX = 22000
# szY = 28000
# resolution = 0.92

def getRegMask(jAtlas, szX, szY, resolution):
    outImg = np.zeros((szX,szY), dtype='uint8')
    # print(file)
    # with open(file) as fAtlas:
    #     jAtlas = json.load(fAtlas)

    perim_coords = np.empty((0,2), dtype='int')

    for atlasRegions in jAtlas['features']:
        if len(atlasRegions['geometry']['coordinates']):
            for regns in (atlasRegions['geometry']['coordinates'][0]):
                if len(np.array(regns).shape) == 2:
                    coords = np.zeros((len(regns), 2), dtype='int')
                    count = 0
                    for ptsRegn in regns:
                        coords[count,0] = np.int((ptsRegn[0]/resolution)+szY/2)
                        coords[count,1] = np.int((ptsRegn[1]/resolution)+szX/2)
                        count = count + 1
                    perim_coords = np.append(perim_coords, coords, axis=0)
                    
                elif len(regns)>2:
                    for subregn in regns:
                        coords = np.zeros((len(subregn),2), dtype='int')
                        count = 0
                        for ptsRegn in subregn:
                            coords[count, 0] = np.int((ptsRegn[0]/resolution)+szY/2)
                            coords[count, 1] = np.int((ptsRegn[1]/resolution)+szX/2)
                            count = count + 1 
                        perim_coords = np.append(perim_coords, coords, axis=0)

                elif len(regns)==2:
                    for subregn in regns:
                        coords = np.zeros((len(subregn), 2), dtype='int')
                        count = 0
                        for ptsRegn in subregn:
                            coords[count, 0] = np.int((ptsRegn[0]/resolution)+szY/2)
                            coords[count, 1] = np.int((ptsRegn[1]/resolution)+szX/2)
                            count = count + 1  
                        perim_coords = np.append(perim_coords, coords, axis=0)

    if len(perim_coords):
        perim_coords = np.asarray(perim_coords)
        cv2.fillConvexPoly(outImg, cv2.convexHull(perim_coords), (255, 255, 255)) 
    else:
        outImg = np.ones((szX,szY), dtype='uint8')*255
    # cv2.imwrite(file.replace('.geojson', '_mask.jp2'), outImg)
    return outImg

# getRegMask(jAtlas, szX, szY, resolution)


