# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:46:26 2019

@author: sheva
"""
import cv2
import numpy as np
import pandas as pd
import yaml
import functions_2D
import os
from matplotlib import pyplot as plt

start_id = 67
end_id = 240
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

Cammat_1,Cammat_2,Proj_1,Proj_2,dist_1,dist_2 = functions_2D.calc_projection(camera_parameter_file_path_L,camera_parameter_file_path_R)
print(Cammat_1,Cammat_2,Proj_1,Proj_2)

true = functions_2D.getTruePose()

idx = start_id
while idx <= end_id:
        
    #Detcet the left image at Tk-1
    imLprev = cv2.imread(os.path.join(images_LC_path, filelist_LC['Imagenumber'][idx]), 0)
    print("processing images",filelist_LC['Imagenumber'][idx],idx)
    #Detcet the left image at Tk
    imLnext = cv2.imread(os.path.join(images_LC_path, filelist_LC['Imagenumber'][idx+1]), 0)
    print("processing images",filelist_LC['Imagenumber'][idx+1],idx+1)
    
    #Detcet the right image at Tk-1
    imRprev = cv2.imread(os.path.join(images_RC_path, filelist_RC['Imagenumber'][idx]), 0)
    print("processing images",filelist_RC['Imagenumber'][idx],idx)
    #Detcet the left image at Tk
    imRnext = cv2.imread(os.path.join(images_RC_path, filelist_RC['Imagenumber'][idx+1]), 0)
    print("processing images",filelist_RC['Imagenumber'][idx+1],idx+1)
    #get key points and descriputor orb
#    kpLprev,descLprev = functions_2D.getFeatures_orb(imLprev)
#    kpLnext,descLnext = functions_2D.getFeatures_orb(imLnext)
#    
#    kpRprev,descRprev = functions_2D.getFeatures_orb(imRprev)
#    kpRnext,descRnext = functions_2D.getFeatures_orb(imRnext)
    
    #get key points and descriputor sift
    kpLprev,descLprev = functions_2D.getFeatures_SIFT(imLprev,size)
    kpLnext,descLnext = functions_2D.getFeatures_SIFT(imLnext,size)

    kpRprev,descRprev = functions_2D.getFeatures_SIFT(imRprev,size)
    kpRnext,descRnext = functions_2D.getFeatures_SIFT(imRnext,size)
    print(descLprev.shape,descLnext.shape)
    
    #get key points and descriputor surf
#    kpLprev,descLprev = functions_2D.getFeatures_SURF(imLprev,size)
#    kpLnext,descLnext = functions_2D.getFeatures_SURF(imLnext,size)
#
#    kpRprev,descRprev = functions_2D.getFeatures_SURF(imRprev,size)
#    kpRnext,descRnext = functions_2D.getFeatures_SURF(imRnext,size)
#    print(kpLprev)
#    print (len(kpLprev))
    ##correspondences
    corLprev, corLnext, matchKeptL_T1,matchKeptL_T2,goodL = functions_2D.getCorres(descLprev,descLnext,kpLprev,kpLnext)
    corRprev, corRnext, matchKeptR_T1,matchKeptR_T2,goodR = functions_2D.getCorres(descRprev,descRnext,kpRprev,kpRnext)
    print('L1 , L2 :', corLprev.shape, corLnext.shape)
    print('R1 , R2 :', corRprev.shape, corRnext.shape)
    
    
    #Debug imgages
#    print(corLprev.T, corLnext.T)
#    
#    plt.figure(figsize=(10,10))
#    plt.plot(corLprev.T[:,0:1],corLprev.T[:,1:2], c='r', label = 'kpL_T1')
#    plt.plot(corLnext.T[:,0:1],corLnext.T[:,1:2], c='g', label = 'kpL_T2')
#    plt.xlim((0,2464))
#    plt.ylim((2056,0))
#    plt.xlabel("$x-direction(pixel)$")
#    plt.ylabel("$y-direction(pixel)$")
#    plt.legend()
    #debug matches:
    img_matches = np.empty((max(imLprev.shape[0], imLnext.shape[0]), imLnext.shape[1]+imLnext.shape[1], 3), dtype=np.uint8)
    img3 = cv2.drawMatches(imLprev, kpLprev, imLnext, kpLnext, goodL, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
#    cv2.namedWindow('Good Matches', 0)
#    cv2.imshow('Good Matches', img3)
#    cv2.waitKey()
#    calculate vehicle motion
    truth_x1, truth_y1, truth_z1, absolute_scale1 = functions_2D.getAbsoluteScale(filelist_LC, idx)
    
    truth_x2, truth_y2, truth_z2, absolute_scale2 = functions_2D.getAbsoluteScale(filelist_RC, idx)
    print(absolute_scale1,absolute_scale2)
#t1-t2
    E1, mask1 = cv2.findEssentialMat(corLprev.T, corLnext.T, Cammat_1, cv2.RANSAC,threshold = 1, prob = .99)
    _1, R1, t1, mask1 = cv2.recoverPose(E1, corLprev.T, corLnext.T,Cammat_1)
    
    E2, mask2 = cv2.findEssentialMat(corRprev.T,corRnext.T,  Cammat_2, cv2.RANSAC,threshold = 1, prob = .99)
    _2, R2, t2, mask2 = cv2.recoverPose(E2, corRprev.T,corRnext.T, Cammat_2)
    
#t2-t1
#    E1, mask1 = cv2.findEssentialMat(corLnext.T, corLprev.T, Cammat_1, cv2.RANSAC,threshold = 1, prob = .99)
#    _1, R1, t1, mask1 = cv2.recoverPose(E1, corLnext.T, corLprev.T, Cammat_1)
#    
#    E2, mask2 = cv2.findEssentialMat(corRnext.T, corRprev.T, Cammat_2, cv2.RANSAC,threshold = 1, prob = .99)
#    _2, R2, t2, mask2 = cv2.recoverPose(E2, corRnext.T, corRprev.T, Cammat_2)
    
    transform1 = functions_2D.makeTransform(R1,t1)
    transform2 = functions_2D.makeTransform(R2,t2)
    
    if idx == start_id:
         if absolute_scale1 > 0.1:
             t0 = np.array([[0,0,0]]).T
             t1_f = t0 + absolute_scale1*R1.dot(t1)
             R1_f = R1.dot(R1)
             pos_1 = t0.T
             pos_1 = np.concatenate((pos_1,t1_f.T),axis =0)
             prevpos_1 = pos_1
            
         if absolute_scale2 > 0.1:  
             t0 = np.array([[0,0,0]]).T
             t2_f = t0 + absolute_scale2*R2.dot(t2)
             R2_f = R2.dot(R2)
             pos_2 = t0.T
             pos_2 = np.concatenate((pos_2,t2_f.T),axis =0)
             prevpos_2 = pos_2
        
    else:
        if absolute_scale1 > 0.1: 
            t1_f = t1_f + absolute_scale1 * R1_f.dot(t1)
            R1_f = R1.dot(R1_f)
            pos_1 = np.concatenate((prevpos_1,t1_f.T) ,axis =0)
            prevpos_1 = pos_1
        if absolute_scale2 > 0.1:  
            t2_f = t2_f + absolute_scale2 * R2_f.dot(t2)
            R2_f = R2.dot(R2_f)
            pos_2 = np.concatenate((prevpos_2,t2_f.T), axis =0)
            prevpos_2 = pos_2
     
#    if idx == start_id:
#         if absolute_scale1 > 0.1:
#             pos1 = np.array([[0,0,0]])
#             newpos1 = functions_2D.posUpdate(pos1,transform1)
#             pos1 = np.concatenate((pos1,newpos1),axis =0)
#             delta1 = newpos1-pos1
#             prevpos1 = newpos1
#            
#         if absolute_scale2 > 0.1:  
#             pos2 = np.array([[0,0,0]])
#             newpos2 = functions_2D.posUpdate(pos2,transform2)
#             pos2 = np.concatenate((pos2,newpos2),axis =0)
#             delta2 = newpos2-pos2
#             prevpos2 = newpos2
#        
#    else:
#        if absolute_scale1 > 0.1: 
#            newpos1 = functions_2D.posUpdate(prevpos1,transform1)
#            pos1 = np.concatenate((pos1,newpos1),axis =0)
#            delta1 = newpos1-prevpos1
#            prevpos1 = newpos1
#        if absolute_scale2 > 0.1:  
#            newpos2 = functions_2D.posUpdate(prevpos2,transform2)
#            pos2 = np.concatenate((pos2,newpos2),axis =0)
#            delta2 = newpos2-prevpos2
#            prevpos2 = newpos1
#    trans = trans1    
##    trans = (trans1+trans2)/2
#    print(trans2)
#    
#    rot = rot1
##    rot = (rot1+rot2)/2
#    print(rot2)
#    print(matchKeptL_St,matchKeptR_St)
#    transform1 = functions_2D.makeTransform(R1_f,t1_f)
#    transform2 = functions_2D.makeTransform(R2_f,t2_f)

    
#    if idx == start_id:
#        pos = np.array([[0,0,0]])
#        newpos = functions_2D.posUpdate(pos,transform1)
#        pos = np.concatenate((pos,newpos),axis =0)
#        prevpos = newpos
#
#    else:
#        newpos = functions_2D.posUpdate(prevpos,transform1)
#        pos = np.concatenate((pos,newpos),axis =0)
#        prevpos = newpos
    print(t1)
    print(pos_1)
#    print(pos1)
#    print(mask1)
    
    idx += 1
plt.figure(figsize=(10,10))
plt.title('2D results with ORB_detector',fontsize = 20)
plt.plot(true[start_id:end_id,1],true[start_id:end_id,0],c ='r',label = 'Dr_data')
plt.plot((pos_1[:,0:1]+true[start_id,1]),(-pos_1[:,1:2]+true[start_id,0]),c= 'g',label = 'LC')
plt.plot((pos_2[:,0:1]+true[start_id,1]),(-pos_2[:,1:2]+true[start_id,0]),c= 'b',label = 'RC')
plt.xlim((205,220))
plt.xlabel("$Easting (m)$",fontsize = 20)
plt.ylabel("$Northing (m)$",fontsize = 20)
plt.legend(fontsize = 20)
