# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 20:59:25 2019

@author: sheva
"""
import datetime
import cv2
import pandas as pd
import numpy as np
import functions_3D
import os
import yaml
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt






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
    print(T[:m, m])
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


def icp(A, B, init_pose=None, max_iterations=10, tolerance=0.001):
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
#    print(m)
    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i






start_id = 0
end_id = 0
idx = start_id
size =[2,2]

with open('D:/Southampton/Msc Project/TEST FOLDER/SVO_configuration.yaml', 'r') as stream:
    load_data_mission = yaml.load(stream)
    
#Load the filelist of left images
filelist_LC_path = load_data_mission.get('Images_L_path', None)
filelist_LC = pd.read_csv(filelist_LC_path)
print(list(filelist_LC))

#Load the filelist of Right images
filelist_RC_path = load_data_mission.get('Images_R_path', None)
filelist_RC = pd.read_csv(filelist_RC_path)

#Load images 
images_LC_path = load_data_mission.get('LC', None)
images_RC_path = load_data_mission.get('RC', None)

#Get camera parameters and calculate projection matirxs
camera_parameter_file_path_L = load_data_mission.get('params_path_LC', None)
camera_parameter_file_path_R = load_data_mission.get('params_path_RC', None)

Cammat_1,Cammat_2,Proj_1,Proj_2,dist_1,dist_2 = functions_3D.calc_projection(camera_parameter_file_path_L,camera_parameter_file_path_R)
print(Cammat_1,Cammat_2,Proj_1,Proj_2)

true = functions_3D.getTruePose()

while idx <= end_id:
    
    #Detcet the image at Tk-1
    imLprev = cv2.imread(os.path.join(images_LC_path, filelist_LC['Imagenumber'][idx]), 0)
    imRprev = cv2.imread(os.path.join(images_RC_path, filelist_RC['Imagenumber'][idx]), 0)
    print("processing images",filelist_LC['Imagenumber'][idx],idx)
    print("processing images",filelist_RC['Imagenumber'][idx],idx)
    #Detcet the image at Tk
    imLnext = cv2.imread(os.path.join(images_LC_path, filelist_LC['Imagenumber'][idx+1]), 0)
    print("processing images",filelist_LC['Imagenumber'][idx+1],idx+1)
    imRnext = cv2.imread(os.path.join(images_RC_path, filelist_RC['Imagenumber'][idx+1]), 0)
    #get key points and descriputor surf_bucket
#    kpLprev,descLprev = functions_3D.getFeatures_SIFT(imLprev,size)
#    kpRprev,descRprev = functions_3D.getFeatures_SIFT(imRprev,size)
#    
#    
#    #Detect the image at Tk_bucket
#    kpLnext,descLnext = functions_3D.getFeatures_SIFT(imLnext,size)
#    kpRnext,descRnext = functions_3D.getFeatures_SIFT(imRnext,size)
    
    #get key points and descriputor orb
    kpLprev,descLprev = functions_3D.getFeatures_ORB(imLprev)
    kpRprev,descRprev = functions_3D.getFeatures_ORB(imRprev)
    
    kpLnext,descLnext = functions_3D.getFeatures_ORB(imLnext)
    kpRnext,descRnext = functions_3D.getFeatures_ORB(imRnext)
    print(descLprev.shape,descRprev.shape)
#    print(kpLprev)
#    print (len(kpLprev))
    ##correspondences
    corLprev_St, corRprev_St,matchKeptL_St,matchKeptR_St,good1 = functions_3D.getmatches(descLprev,descRprev,kpLprev,kpRprev,1)
    print(corLprev_St.shape,corRprev_St.shape)
    corLnext_St, corRnext_St,matchKeptL_Ed,matchKeptR_Ed,good2 = functions_3D.getmatches(descLnext,descRnext,kpLnext,kpRnext,1)
    
    
    
    #Triangulate points
    x3dprev = (functions_3D.triangulate(corLprev_St.T, corRprev_St.T, Proj_1,Proj_2)).T
    x3dnext = (functions_3D.triangulate(corLnext_St.T, corRnext_St.T, Proj_1,Proj_2)).T
    print(x3dprev.shape)
    x3dprev = x3dprev[:,:3]
    x3dnext = x3dnext[:,:3]
    
    if len(x3dprev) >= len(x3dnext):
        x3dprev = x3dprev[:len(x3dnext),:]
    elif len(x3dprev) < len(x3dnext):
        x3dnext = x3dnext[:len(x3dprev),:]
    print(x3dprev,x3dnext)

    M1_2, mask1_2 = cv2.findHomography(x3dprev, x3dnext, cv2.RANSAC,0.5)
    mask1_2 = np.squeeze(mask1_2,axis =1)
#
    x3dprev = x3dprev[mask1_2==1]
    x3dnext = x3dnext[mask1_2==1]    
    transform,distances,iteration = functions_3D.icp(x3dprev,x3dnext)
    print(transform)
    
    
    if idx ==start_id:
        pos = np.array([[0,0,0]])
        newpos = functions_3D.posUpdate(pos,transform)
        pos = np.concatenate((pos,newpos),axis =0)
        delta = newpos-pos
        prevpos = newpos
        

    else:
        newpos = functions_3D.posUpdate(prevpos,transform)
        pos = np.concatenate((pos,newpos),axis =0)
        delta = newpos-prevpos
        prevpos = newpos
        
    
    print(delta)
    print(pos)
    
    idx += 1
    
#    print(matchKeptL_St,matchKeptR_St)
#    img_matches = np.empty((max(imLprev.shape[0], imRprev.shape[0]), imLprev.shape[1]+imRprev.shape[1], 3), dtype=np.uint8)
#    img3 = cv2.drawMatches(imLprev, kpLprev, imRprev, kpRprev, good1, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#    print((x3dprev),(pErrPrev),(matchKept_3d))
    
#    Test matches
#    cv2.namedWindow('Good Matches', 0)
#    cv2.imshow('Good Matches', img3)
#    cv2.waitKey()
    
#    kpLnext,descLnext = functions_3D.getFeatures_ORB(imLnext)


