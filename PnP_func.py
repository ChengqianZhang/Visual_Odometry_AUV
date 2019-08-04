# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:04:09 2019

@author: sheva
"""

# Main helper functions for stereo visual odometry

import numpy as np
import cv2
import math
import operator

#kp1_T1, des1_T1 = detector.detectAndCompute(imgT1_L, None)
#kp2_T1, des2_T1 = detector.detectAndCompute(imgT1_R, None)
#kp1_T2, des1_T2 = detector.detectAndCompute(imgT2_L, None)
#kp2_T2, des2_T2 = detector.detectAndCompute(imgT2_R, None)


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
   
    
def getTruePose():
    file = 'D:/Southampton/Msc Project/TEST FOLDER/truepose.txt'
    return np.genfromtxt(file, delimiter=' ',dtype=None)
#Note: in 'bucket_PnP.py' there is another dectector use feature bucketing


def getFeatures_SURF(img):

    surf = cv2.xfeatures2d.SURF_create(
            hessianThreshold=100,
            nOctaves = 4,
            nOctaveLayers = 3)
    kp, desc = surf.detectAndCompute(img,None)
    return kp,desc


def getFeatures_SIFT(img):
    sift = cv2.xfeatures2d.SIFT_create(
            nfeatures = 500, 
            nOctaveLayers = 3,
            contrastThreshold =0.035, 
            edgeThreshold = 30)
    kp, desc = sift.detectAndCompute(img, None)
    returns = (kp,desc)
    return returns
    
    



def bucket(img,size):
    H = size[0]
    W = size[1]
    print(H,W)
    
    height, width= img.shape
    sift = cv2.xfeatures2d.SIFT_create(nfeatures = 100, 
            nOctaveLayers = 3,
            contrastThreshold =0.02, 
            edgeThreshold = 40)
    
    kp = []
    for y in range(0, height,H):
        for x in range(0, width,W):
            imPatch = img[y:y+H, x:x+W]
            kpTemp, desctemp = sift.detectAndCompute(imPatch,None)
            print(desctemp)
            for idx in range(len(kpTemp)):
                # Compensate for global offset
                kpTemp[idx].pt = tuple(map(operator.add,  kpTemp[idx].pt, (x,y)))
            if x ==0 and y ==0:
                desc = desctemp
            else:
                desc = np.concatenate((desc,desctemp), axis=0)
            kp=kp+kpTemp
    return kp,desc
                
def getFeatures_SIFT_buck(img, sizeWindowArray ):
    
    height, width= img.shape

    xstep = int(width/sizeWindowArray[1])
    ystep = int(height/sizeWindowArray[0])
    print(xstep,ystep)
#    print(xstep,ystep)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures = 1000, 
            nOctaveLayers = 3,
            contrastThreshold =0.02, 
            edgeThreshold = 40)
    
    kp = []
    # Loop through windows
    for y in range(0,ystep*sizeWindowArray[0],ystep):

        for x in range(0,xstep*sizeWindowArray[1],xstep):
            
            # Find SIFT features
            kpTemp, desctemp = sift.detectAndCompute(img[y:y+ystep-1,x:x+xstep-1],None)
            print(desctemp.shape)
            
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
def getFeatures_ORB(img):
    
    detector = cv2.ORB_create(
            nfeatures=1500,
            scaleFactor=1.2,
            nlevels= 8,
            edgeThreshold= 35,
            firstLevel= 0,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20)
#    detector = cv2.ORB_create(10000)
    kp, desc = detector.detectAndCompute(img, None)
    returns = (kp,desc)
    return returns


def getCorres(desc1, desc2, kp1, kp2,ratio):

    k = 2

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



def getCorres_2(desc1, desc2, kp1, kp2):


# Brute force with ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1,desc2)
    
    matches = sorted(matches, key = lambda x:x.distance)
    
    good = []
    for m in matches:
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
      x_pre, y_pre, z_pre = f[' Northing [m]'][idx-1], f[' Easting [m]'][idx-1], f[' Altitude [m]'][idx-1]
      x    , y    , z     = f[' Northing [m]'][idx], f[' Easting [m]'][idx], f[' Altitude [m]'][idx]
      scale = np.sqrt((x-x_pre)**2 + (y-y_pre)**2 + (z-z_pre)**2)
      return x, y, z, scale



def triangulate(x2dL,x2dR,PL,PR):
    #x2d's must be of size 2 x N
#    PL = np.matrix('2.46385135e+03 0.00000000e+00 1.18586256e+03 0.00000000e+00; 0.00000000e+00 2.46385135e+03 9.73449051e+02 0.00000000e+00; 0.00000000e+00 0.00000000e+00 1.00000000e+00 0.00000000e+00')
#    PR = np.matrix('2.46385135e+03 0.00000000e+00 1.18586256e+03 -3.69223072e+02; 0.00000000e+00 2.46385135e+03 9.73449051e+02 0.00000000e+00; 0.00000000e+00 0.00000000e+00 1.00000000e+00 0.00000000e+00')
    x3d = cv2.triangulatePoints(PL,PR,x2dL,x2dR) #x3d will be of size 4 x N
	
    x3d /= x3d[3] #forcing homogeneity

    x3d = np.matrix(x3d)

	#calculating reprojection error between collected feature points and the reprojected 3d coord into 2d
	#matrix multiplication
    phatL = PL*x3d
    phatR = PR*x3d

    phatL /= phatL[2]
    phatR /= phatR[2] 

    x3d=x3d[:3]
    return x3d

    


def camPose(x3d,x2dL,cameramat):
##convert from matrix to array
#    x3d_PnP=np.matrix(x3d)
#    x2dL_PnP=np.matrix(x2dL)
    	#import pdb;pdb.set_trace() #for debugging
    dist = np.zeros((5,1))
    _, rvec, tvec, inliers = cv2.solvePnPRansac(x3d, x2dL, cameramat,dist,iterationsCount = 1000, confidence = 0.99, flags = cv2.SOLVEPNP_ITERATIVE)#, flags = cv2.SOLVEPNP_ITERATIVE) #useExtrinsicGuess = False, iterationsCount = 50, reprojectionError = 20, confidence = 0.95, flags = cv2.SOLVEPNP_ITERATIVE)
    #retrieve the rotation matrix
    rot,_ = cv2.Rodrigues(rvec)
#    tvec = rot.dot(tvec)      #tvec = -rot.T.dot(tvec)  #coordinate transformation, from camera to worl
    return rot,tvec
	#return (rotRect,transRect)
def bundleAdjust(perror,rot,trans):
	return 0                    



def find_overlap(idxprevL,idxnextL):
    pt1 = np.array(idxprevL).T
    pt2 = np.array(idxnextL).T
    
    pt1 = pt1.ravel().tolist()
    pt2 = pt2.ravel().tolist()
    pt3 = [value for value in pt1 if value in pt2]
    idx1 = 0
    idx2 = 0
    mask = []
    while idx1 <= (len(pt1)-1) and idx2 <= (len(pt3)-1):
        if pt1[idx1] == pt3[idx2]:
            mask.append(1)
            idx1 += 1
            idx2 += 1
        else:
            mask.append(0)
            idx1 += 1
    while idx1 <= (len(pt1)-1):
        mask.append(0)
        idx1 += 1      
    return np.array(pt3),np.array(mask)

def find_index(idxnextL,overlapL):
    pt1 = np.array(idxnextL).T
    
    pt1 = pt1.ravel().tolist()
    pt2 = overlapL
    pt2 = pt2.ravel().tolist()
    idx1 = 0
    idx2 = 0
    mask = []
    while idx1 <= (len(pt1)-1) and idx2 <= (len(pt2)-1):
        if pt1[idx1] == pt2[idx2]:
            mask.append(1)
            idx1 += 1
            idx2 += 1
        else:
            mask.append(0)
            idx1 += 1
    while idx1 <= (len(pt1)-1):
        mask.append(0)
        idx1 += 1      
    return np.array(mask)

#def flatten(items):
#    """Yield items from any nested iterable; see Reference."""
#    for x in items:
#        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
#            for sub_x in flatten(x):
#                yield sub_x
#        else:
#            yield x


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

#The body to inertial transformation
def body_to_inertial(roll, pitch, yaw, old_x, old_y, old_z):
    deg_to_rad = math.pi/180
    roll = roll*deg_to_rad
    pitch = pitch*deg_to_rad
    yaw = yaw*deg_to_rad
    new_x = ((math.cos(yaw)*math.cos(pitch))*old_x+(-math.sin(yaw)*math.cos(roll)+math.cos(yaw)*math.sin(pitch)*math.sin(roll))*old_y+(math.sin(yaw)*math.sin(roll)+(math.cos(yaw)*math.cos(roll)*math.sin(pitch)))*old_z)
    new_y = ((math.sin(yaw)*math.cos(pitch))*old_x +(math.cos(yaw)*math.cos(roll)+math.sin(roll)*math.sin(pitch)*math.sin(yaw))*old_y+(-math.cos(yaw)*math.sin(roll)+math.sin(yaw)*math.cos(roll)*math.sin(pitch))*old_z)
    new_z = ((-math.sin(pitch)*old_x)+(math.cos(pitch)*math.sin(roll)) * old_y+(math.cos(pitch)*math.cos(roll))*old_z)
    return new_x,new_y,new_z





#
#LC,RC,LCIDX,RCIDX,good_T1 = getCorres(des1_T1,des2_T1,kp1_T1,kp2_T1)
#print(LC.T,RC)
#print(LCIDX)
#print(good_T1)
#img_matches = np.empty((max(imgT1_L.shape[0], imgT1_R.shape[0]), imgT1_L.shape[1]+imgT1_R.shape[1], 3), dtype=np.uint8)
##img3 = cv2.drawMatches(imgT1_L, kp1_T1, imgT1_R, kp2_T1, good_T1, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#good_T1 = list(flatten(good_T1))
#img4 = cv2.drawMatches(imgT1_L, kp1_T1, imgT1_R, kp2_T1,  good_T1, None)
#cv2.namedWindow('Good Matches', 0)
#cv2.imshow('Good Matches', img4)
#cv2.waitKey()