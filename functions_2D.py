# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:02:57 2019

@author: sheva
"""

import cv2
import numpy as np
import operator

def calc_projection(camera_parameter_file_path_L,camera_parameter_file_path_R):
    fl = cv2.FileStorage(camera_parameter_file_path_L, cv2.FILE_STORAGE_READ)
    fr = cv2.FileStorage(camera_parameter_file_path_R, cv2.FILE_STORAGE_READ)
    fn = fl.getNode('camera_matrix')
    camera_matrix_1 = fn.mat()
    fn = fl.getNode('distortion_coefficients')
    distortion_coefficients_1 = fn.mat()
    fn = fl.getNode('Rotation')
    rotation_1 = fn.mat()
    fn = fl.getNode('Translation')
    T_1 = fn.mat()
    #Right camera
    fn = fr.getNode('camera_matrix')
    camera_matrix_2 = fn.mat()
    fn = fr.getNode('distortion_coefficients')
    distortion_coefficients_2 = fn.mat()
    fn = fr.getNode('Rotation')
    rotation_2 = fn.mat()
    fn = fr.getNode('Translation')
    T_2 = fn.mat()
    
    #Calculate projection
    RT_1 = np.concatenate((rotation_1,T_1),axis=1)
    RT_2 = np.concatenate((rotation_2,T_2),axis=1)
    proj1 = camera_matrix_1 @ RT_1
    proj2 = camera_matrix_2 @ RT_2
    
    return camera_matrix_1, camera_matrix_2, proj1, proj2, distortion_coefficients_1, distortion_coefficients_2


def getFeatures_orb(img):
    
    detector = cv2.ORB_create(1500)
    kp, desc = detector.detectAndCompute(img, None)
    returns = (kp,desc)
    return returns

def getFeatures_SIFT(img, sizeWindowArray ):
    
    height, width= img.shape

    xstep = int(width/sizeWindowArray[1])
    ystep = int(height/sizeWindowArray[0])
    sift = cv2.xfeatures2d.SIFT_create(1000)
    
#    desc = np.zeros(1)
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

def getFeatures_SURF(img, sizeWindowArray ):
    
    height, width= img.shape

    xstep = int(width/sizeWindowArray[1])
    ystep = int(height/sizeWindowArray[0])
    minHessian = 180
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=minHessian)
    
#    desc = np.zeros(1)
    kp = []
    # Loop through windows
    for y in range(0,ystep*sizeWindowArray[0],ystep):

        for x in range(0,xstep*sizeWindowArray[1],xstep):
            
            # Find SIFT features
            kpTemp, desctemp = surf.detectAndCompute(img[y:y+ystep-1,x:x+xstep-1],None)
            
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
def getCorres(desc1, desc2, kp1, kp2):

    k = 2
    ratio = 0.71

# Brute force with ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k)
    
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)

    src = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2,1)
    dst = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2,1)
    M, mask = cv2.findHomography(src, dst, cv2.RANSAC,5.0)
    mask = mask.ravel().tolist()
    matches = dict(zip(good, mask))
    matches = {k:v for k,v in  matches.items() if v != 0}
    good_matches = list(matches.keys())

    
    leftCorres = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,2,1)
    rightCorres = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,2,1)
#    import pdb; pdb.set_trace()
   # good of type Dmatch 
    leftCorresidx = np.matrix([good_matches[m].queryIdx for m in range(len(good_matches)) ])
    rightCorresidx = np.matrix([ good_matches[m].trainIdx for m in range(len(good_matches)) ])

    leftCorres = np.matrix.transpose(np.concatenate((leftCorres[:,0],leftCorres[:,1]),axis=1))
    rightCorres = np.matrix.transpose(np.concatenate((rightCorres[:,0],rightCorres[:,1]),axis=1))

    returns = (leftCorres, rightCorres, leftCorresidx, rightCorresidx,good_matches)
    return returns

def getAbsoluteScale(f, idx):
      x_pre, y_pre, z_pre = f[' Northing [m]'][idx], f[' Easting [m]'][idx], f[' Altitude [m]'][idx]
      x    , y    , z     = f[' Northing [m]'][idx+1], f[' Easting [m]'][idx+1], f[' Altitude [m]'][idx+1]
      scale = np.sqrt((x-x_pre)**2 + (y-y_pre)**2 + (z-z_pre)**2)
      return x, y, z, scale

def getTruePose():
    file = 'D:/Southampton/Msc Project/TEST FOLDER/truepose.txt'
    return np.genfromtxt(file, delimiter=' ',dtype=None) 
def makeTransform(rot,trans):
    transform = np.concatenate((rot,trans), axis=1)
    transform = np.asmatrix(transform)
    lastRow = np.matrix([0,0,0,1])
    transform = np.concatenate((transform,lastRow), axis=0)

    return transform

def posUpdate(pos, transform):
    # input array, output matrix
#    import pdb;pdb.set_trace()
    pos =  np.asmatrix(pos)
    pos = pos.T
    one = np.matrix(1)
    pos = np.concatenate((pos,one),axis=0)
    newpos = transform*pos
    newpos = newpos[0:3,:]
    newpos = newpos.T
    newpos = np.asarray(newpos)
#    newpos = newpos[0]
#   output array
    return (newpos)
