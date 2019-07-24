# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:04:09 2019

@author: sheva
"""

import numpy as np
import cv2
import operator


#kp1_T1, des1_T1 = detector.detectAndCompute(imgT1_L, None)
#kp2_T1, des2_T1 = detector.detectAndCompute(imgT1_R, None)
#kp1_T2, des1_T2 = detector.detectAndCompute(imgT2_L, None)
#kp2_T2, des2_T2 = detector.detectAndCompute(imgT2_R, None)
def getFeatures(img, sizeWindowArray ):
    
    height, width= img.shape

    xstep = int(width/sizeWindowArray[1])
    ystep = int(height/sizeWindowArray[0])
    sift = cv2.xfeatures2d.SIFT_create(1000)
    
    kp = []
    # Loop through windows
    for y in range(0,ystep*sizeWindowArray[0],ystep):

        for x in range(0,xstep*sizeWindowArray[1],xstep):
            
            # Find SIFT features
            kpTemp, desctemp = sift.detectAndCompute(img[y:y+ystep-1,x:x+xstep-1],None)
            
            for idx in range(len(kpTemp)):
                # Compensate for global offset
                kpTemp[idx].pt = tuple(map(operator.add,  kpTemp[idx].pt, (x,y)))

            if x==0 and y==0:
                desc = desctemp
            else:
                desc = np.concatenate((desc,desctemp), axis=0)

            kp = kp + kpTemp
    
    returns = (kp,desc)
    return returns