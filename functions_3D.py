# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 20:47:47 2019

@author: sheva
"""

import numpy as np
import cv2
import math
import operator
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    print(R,t)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=80, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i


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

def getFeatures_SIFT(img, sizeWindowArray ):
    
    height, width= img.shape

    xstep = int(width/sizeWindowArray[1])
    ystep = int(height/sizeWindowArray[0])
    sift = cv2.xfeatures2d.SIFT_create(400)
    
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


def getFeatures_ORB(img):
    detector = cv2.ORB_create(1000)
    kp, desc = detector.detectAndCompute(img, None)
    returns = (kp,desc)
    return returns

def getCorres(desc1, desc2, kp1, kp2,ratio):

    k = 2

# Brute force with ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k)
#    matches = sorted(matches, key = lambda x:x.distance)
    
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

    
    leftCorres = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,2)
    rightCorres = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,2)
#    import pdb; pdb.set_trace()
   # good of type Dmatch 
    leftCorresidx = np.matrix([good_matches[m].queryIdx for m in range(len(good_matches)) ])
    rightCorresidx = np.matrix([ good_matches[m].trainIdx for m in range(len(good_matches)) ])
#
#    leftCorres = np.matrix.transpose(np.concatenate((leftCorres[:,0],leftCorres[:,1]),axis=1))
#    rightCorres = np.matrix.transpose(np.concatenate((rightCorres[:,0],rightCorres[:,1]),axis=1))

    returns = (leftCorres, rightCorres, leftCorresidx, rightCorresidx,good_matches)
    return returns



def getmatches(desc1, desc2, kp1, kp2,ratio):

    k = 2

# Brute force with ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k)
#    matches = sorted(matches, key = lambda x:x.distance)
    
    good = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good.append(m)

    src = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src, dst, cv2.RANSAC,5.0)
    mask = mask.ravel().tolist()
    matches = dict(zip(good, mask))
    matches = {k:v for k,v in  matches.items() if v != 0} 
    good_matches = list(matches.keys())

    
    leftCorres = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,2)
    rightCorres = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,2)
#    import pdb; pdb.set_trace()
   # good of type Dmatch 
    leftCorresidx = np.matrix([good_matches[m].queryIdx for m in range(len(good_matches)) ])
    rightCorresidx = np.matrix([ good_matches[m].trainIdx for m in range(len(good_matches)) ])
#
#    leftCorres = np.matrix.transpose(np.concatenate((leftCorres[:,0],leftCorres[:,1]),axis=1))
#    rightCorres = np.matrix.transpose(np.concatenate((rightCorres[:,0],rightCorres[:,1]),axis=1))

    return leftCorres, rightCorres, leftCorresidx, rightCorresidx,good_matches



def triangulate(x2dL,x2dR,PL,PR):
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

def camPose(x3d_T1,x3d_T2):
    
    T, distances, i = icp(x3d_T1,x3d_T2)  
#    tvec = rot.dot(tvec)      #tvec = -rot.T.dot(tvec)  #coordinate transformation, from camera to worl
    return T,i



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