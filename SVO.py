# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:53:43 2019

@author: sheva
"""
import datetime
import cv2
import pandas as pd
import numpy as np
import PnP_func
import os
import yaml
from matplotlib import pyplot as plt
import bucket_PnP

numpics = 23
with open('D:/Southampton/Msc Project/TEST FOLDER/SVO_configuration.yaml', 'r') as stream:
    load_data_mission = yaml.load(stream)
    
#Load the filelist of left images (.csv) format
filelist_LC_path = load_data_mission.get('Images_L_path', None)
filelist_LC = pd.read_csv(filelist_LC_path)
print(list(filelist_LC))

#Load the filelist of Right images
filelist_RC_path = load_data_mission.get('Images_R_path', None)
filelist_RC = pd.read_csv(filelist_RC_path)

#Load images 
images_LC_path = load_data_mission.get('LC', None)
images_RC_path = load_data_mission.get('RC', None)
#Process all the images
idx = 23
size =[2,2]
while idx <= numpics:
    
    #Detcet the image at Tk-1
    imLprev = cv2.imread(os.path.join(images_LC_path, filelist_LC['Imagenumber'][idx]), 0)
    imRprev = cv2.imread(os.path.join(images_RC_path, filelist_RC['Imagenumber'][idx]), 0)
    print("processing images",filelist_LC['Imagenumber'][idx],idx)
    print("processing images",filelist_RC['Imagenumber'][idx],idx)
    #Detcet the image at Tk
    imLnext = cv2.imread(os.path.join(images_LC_path, filelist_LC['Imagenumber'][idx+1]), 0)
    print("processing images",filelist_LC['Imagenumber'][idx+1],idx+1)
    imRnext = cv2.imread(os.path.join(images_RC_path, filelist_RC['Imagenumber'][idx+1]), 0)
    #get key points and descriputor
    kpLprev,descLprev = bucket_PnP.getFeatures(imLprev,size)
    kpRprev,descRprev = bucket_PnP.getFeatures(imRprev,size)
#    print(kpLprev)
#    print (len(kpLprev))
    ##correspondences
    corLprev_St, corRprev_St,matchKeptL_St,matchKeptR_St,good1 = PnP_func.getCorres(descLprev,descRprev,kpLprev,kpRprev)
#    print(matchKeptL_St,matchKeptR_St)
    img_matches = np.empty((max(imLprev.shape[0], imRprev.shape[0]), imLprev.shape[1]+imRprev.shape[1], 3), dtype=np.uint8)
    img3 = cv2.drawMatches(imLprev, kpLprev, imRprev, kpRprev, good1, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#    print((x3dprev),(pErrPrev),(matchKept_3d))
    
    #Test matches
    cv2.namedWindow('Good Matches', 0)
    cv2.imshow('Good Matches', img3)
    cv2.waitKey()
    
    #Detect the image at Tk
    kpLnext,descLnext = bucket_PnP.getFeatures(imLnext,size)
    corLprev_T,corLnext_T,matchKeptPrev_T,matchKeptNext_T,good2 = PnP_func.getCorres(descLprev,descLnext,kpLprev,kpLnext)
    print(corLprev_T.shape,corLnext_T.shape)
    
    
    operlap_L1 ,Mask1 = PnP_func.find_overlap(matchKeptL_St,matchKeptPrev_T)
    Mask2 = PnP_func.find_index(matchKeptPrev_T,operlap_L1)
#    print(operlap_L1,Mask1)
#    print('Mask2 :')
#    print(Mask2)
    
    matchKeptR_St = (np.array(matchKeptR_St).T[Mask1==1]).reshape(-1)
    matchKeptNext_T = (np.array(matchKeptNext_T).T[Mask2==1]).reshape(-1)
#    print(matchKeptR_St)
    
    #Get matched kp from L1 R1 L2
    kpLprev =np.array(kpLprev)[operlap_L1]
#    print(kpLprev)
    kpRprev = np.array(kpRprev)[matchKeptR_St]
    
#    print(kpRprev)
    kpLnext =np.array(kpLnext)[matchKeptNext_T]
#    print(kpLnext)
    
    kpLprev =cv2.KeyPoint_convert(kpLprev)
    kpRprev  =cv2.KeyPoint_convert(kpRprev)
    kpLnext =cv2.KeyPoint_convert(kpLnext)
    
    plt.figure(figsize=(10,10))
    plt.scatter(kpLprev[:,0:1],kpLprev[:,1:2], c='r',label = 'kpL_T1')
    plt.scatter(kpRprev[:,0:1],kpRprev[:,1:2], c='b',label = 'kpR_T1')
    plt.scatter(kpLnext[:,0:1],kpLnext[:,1:2], c='g' ,label = 'kpL_T2')
    plt.xlim((0,2464))
    plt.ylim((2056,0))
    plt.xlabel("$x-direction(pixel)$")
    plt.ylabel("$y-direction(pixel)$")
    plt.legend()
#    
    

    #triangulate must be of the form 2 x N
    x3dprev = (PnP_func.triangulate(kpLprev.T,kpRprev.T)).T
    x3dprev = np.expand_dims(x3dprev,axis =2 )
    kpLprev =np.expand_dims(kpLprev,axis =2 )
    kpLnext = np.expand_dims(kpLnext,axis =2 )
    
    #If the number of the 3d and corresponding 2d points are not enough, try the 2d-2d method for this image sequence
    if len(x3dprev) <= 10 and len(kpLnext) <= 10:
        # Get ground truth and scale
        truth_x, truth_y, truth_z, absolute_scale = PnP_func.getAbsoluteScale(filelist_LC, idx)
        
        cameramat = np.array([[2465.715207123899, 0, 1202.929583462877], 
                              [0, 2466.799723162185, 986.9289744659511], 
                              [0,0,1]])
        fc = 2465.715207123899
        pp = (1202.929583462877, 986.9289744659511)
        E, mask = cv2.findEssentialMat(corLnext_T.T, corLprev_T.T, fc, pp, cv2.RANSAC,0.999,1.0)
        _, R, t, mask = cv2.recoverPose(E, corLnext_T.T, corLprev_T.T,focal=2465.715207123899, pp = pp)
        if absolute_scale > 0.1:  
            trans = t + absolute_scale*R.dot(t)
            rot = R.dot(R)
    else: 
        rot,trans=PnP_func.camPose(x3dprev,kpLnext)
#    print(rot,trans)
        
        
    transform = PnP_func.makeTransform(rot,trans)
#    print(transform)
    
    if idx ==23:
        
        x3d_Saved=(x3dprev)
        pos = np.array([[0,0,0]])
        newpos = PnP_func.posUpdate(pos,transform)
        pos = np.concatenate((pos,newpos),axis =0)
        prevpos = newpos

    else:
        x3d_Saved=np.concatenate((x3d_Saved,x3dprev),axis=0)
        newpos = PnP_func.posUpdate(prevpos,transform)
        pos = np.concatenate((pos,newpos),axis =0)
        prevpos = newpos
    print(pos)

    idx += 1
