# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 20:59:25 2019

@author: sheva
"""
import time
import cv2
import pandas as pd
import numpy as np
import functions_3D
import os
import yaml
import math
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import csv





start_id = 220
end_id = 240
idx = start_id
with open('D:/Southampton/Msc Project/TEST FOLDER/SVO_configuration.yaml', 'r') as stream:
    load_data_mission = yaml.load(stream)
    
#Load the filelist of left images
filelist_LC_path = load_data_mission.get('Images_L_path', None)
filelist_LC = pd.read_csv(filelist_LC_path)
print(list(filelist_LC))

size =[2,2]

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

truePose = functions_3D.getTruePose()
#print(truePose[idx,0])

transx = np.ones((end_id-start_id+1,1))
transy = np.ones((end_id-start_id+1,1))
transz = np.ones((end_id-start_id+1,1))
ERROR = np.zeros((end_id-start_id+1,1))
Altitude = np.ones((end_id-start_id+1,1))
ROT = np.ones((end_id-start_id+1,1))
HEADING = np.ones((end_id-start_id+1,1))
CPD_T = np.ones((end_id-start_id+1,2))
pointset=np.ones((end_id-start_id+1,2))

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
#    kpLprev,descLprev = functions_3D.getFeatures_SIFT(imLprev)
#    kpRprev,descRprev = functions_3D.getFeatures_SIFT(imRprev)
#
#
#    #Detect the image at Tk_bucket
#    kpLnext,descLnext = functions_3D.getFeatures_SIFT(imLnext)
#    kpRnext,descRnext = functions_3D.getFeatures_SIFT(imRnext)
    
    #get key points and descriputor orb
    kpLprev,descLprev = functions_3D.getFeatures_ORB(imLprev)
    kpRprev,descRprev = functions_3D.getFeatures_ORB(imRprev)
    
    kpLnext,descLnext = functions_3D.getFeatures_ORB(imLnext)
    kpRnext,descRnext = functions_3D.getFeatures_ORB(imRnext)
#    print(descLprev.shape,descLnext.shape)
    print(descLprev.shape,descRprev.shape)
    print(descLnext.shape,descRnext.shape)
#    print (len(kpLprev))
    ##correspondences
    
    
    
    p1_L,p2_L,matchKeptPrev_T,matchKeptNext_T,good2 = functions_3D.getCorres(descLprev,descLnext,kpLprev,kpLnext,1)
#    print(matchKeptPrev_T,matchKeptNext_T)
#    p1_L,p1_R,st1 = functions_3D.KLT(imLprev,imRprev,p1_L)
#   #Align matchKeptR_T1
#    print('Left matching', p1_L.shape[0])
    coef = np.array(matchKeptNext_T).ravel()
    coef = coef.tolist()
    coef = sorted(coef)
    matchKeptNext_T=np.matrix([coef])
#    print(matchKeptNext_T)
    

    
    corLprev_St, corRprev_St,matchKeptL_T1,matchKeptR_T1,good1 = functions_3D.getCorres(descLprev,descRprev,kpLprev,kpRprev,1)
#    3d_1 = functions_3D.triangulate(a,a,b,b)
#    print(matchKeptL_T1,matchKeptR_T1)
    
    x3d1 = (functions_3D.triangulate(corLprev_St.T, corRprev_St.T, Proj_1,Proj_2)).T
    altitude, altitude_std = np.mean(np.array(x3d1[:,2])) ,np.std(np.array(x3d1[:,2]))
    print(altitude,altitude_std)
    if abs(altitude) >3 or abs(altitude) <1:
        altitude = Altitude[idx-start_id-1]
        Altitude[idx-start_id] = altitude
    else:
        Altitude[idx-start_id] = altitude
        
    
    corLnext, corRnext, matchKeptL_T2,matchKeptR_T2,good3 = functions_3D.getCorres(descLnext,descRnext,kpLnext,kpRnext,1)
#    print(matchKeptL_T2,matchKeptR_T2)

    #find Overlap
    operlap_T1 ,Mask1 = functions_3D.find_overlap(matchKeptL_T1, matchKeptPrev_T)
    
    operlap_T2 ,Mask2 = functions_3D.find_overlap(matchKeptL_T2, matchKeptNext_T)
#    print(operlap_T1,operlap_T2)
    if len(operlap_T1) >0 and len(operlap_T2)>0:
        matchKeptR_T1 = (np.array(matchKeptR_T1).T[Mask1==1]).reshape(-1)
        matchKeptR_T2 = (np.array(matchKeptR_T2).T[Mask2==1]).reshape(-1)
    
        kpL_t1 = np.array(kpLprev)[operlap_T1]
        kpR_t1 = np.array(kpRprev)[matchKeptR_T1]
        kpL_t2 = np.array(kpLnext)[operlap_T2]
        kpR_t2 = np.array(kpRnext)[matchKeptR_T2]
    
        kpL_t1 = cv2.KeyPoint_convert(kpL_t1)
        kpR_t1 = cv2.KeyPoint_convert(kpR_t1)
        kpL_t2 = cv2.KeyPoint_convert(kpL_t2)
        kpR_t2 = cv2.KeyPoint_convert(kpR_t2)
        x3dprev = (functions_3D.triangulate(kpL_t1.T, kpR_t1.T, Proj_1,Proj_2)).T
        x3dnext = (functions_3D.triangulate(kpL_t2.T, kpR_t2.T, Proj_1,Proj_2)).T
    else:
        x3dprev = (functions_3D.triangulate(corLprev_St.T, corRprev_St.T, Proj_1,Proj_2)).T
        x3dnext = (functions_3D.triangulate(corLnext.T, corRnext.T, Proj_1,Proj_2)).T
        
    
#    plt.figure(figsize=(10,10))
#    plt.scatter(kpL_t1[:,0:1],kpL_t1[:,1:2], c='r',label = 'kpL_T1')
#    plt.scatter(kpR_t1[:,0:1],kpR_t1[:,1:2], c='b',label = 'kpR_T1')
#    plt.scatter(kpL_t2[:,0:1],kpL_t2[:,1:2], c='g',label = 'kpL_T2')
#    plt.scatter(kpR_t2[:,0:1],kpR_t2[:,1:2], c='k',label = 'kpR_T2')
#    plt.xlim((0,2464))
#    plt.ylim((2056,0))
#    plt.xlabel("$x-direction(pixel)$")
#    plt.ylabel("$y-direction(pixel)$")
#    plt.legend()
#    print(corLnext.shape,corRnext.shape)
##   calculate the mean translation between lk+1-rk+1
#    delta_u = np.mean(corRnext[:,0]- corLnext[:,0])
#    delta_v = np.mean(corRnext[:,1]- corLnext[:,1])
#    print(delta_u,delta_v)
#    print(corLprev_T.shape,corLnext_T.shape)
    

    
#    operlap_L1 ,Mask1 = functions_3D.find_overlap(matchKeptL_St,matchKeptPrev_T)
#    Mask2 = functions_3D.find_index(matchKeptPrev_T,operlap_L1)
##    print(operlap_L1,Mask1)
##    print('Mask2 :')
##    print(Mask2)
#    
#    matchKeptR_St = (np.array(matchKeptR_St).T[Mask1==1]).reshape(-1)
#    matchKeptNext_T = (np.array(matchKeptNext_T).T[Mask2==1]).reshape(-1)
##    print(matchKeptR_St)
#    
#    #Get matched kp from L1 R1 L2
#    kpLprev =np.array(kpLprev)[operlap_L1]
##    print(kpLprev)
#    kpRprev = np.array(kpRprev)[matchKeptR_St]
#    
##    print(kpRprev)
#    kpLnext =np.array(kpLnext)[matchKeptNext_T]
##    print(kpLnext)
#    
#    kpLprev =cv2.KeyPoint_convert(kpLprev)
#    kpRprev  =cv2.KeyPoint_convert(kpRprev)
#    kpLnext =cv2.KeyPoint_convert(kpLnext)
##    print(kpLnext.shape)
#    delta_u = np.ones(kpLnext.shape[0])*delta_u
#    delta_v = np.ones(kpLnext.shape[0])*delta_v
##    print(delta_u)
#    kpRnext = np.array([kpLnext[:,0]+delta_u.T,kpLnext[:,1]+delta_v.T]).T
#    print(kpRnext.shape)
    
    
#    plt.figure(figsize=(10,10))
#    plt.scatter(kpLprev[:,0:1],kpLprev[:,1:2], c='r',label = 'kpL_T1')
#    plt.scatter(kpRprev[:,0:1],kpRprev[:,1:2], c='b',label = 'kpR_T1')
#    plt.scatter(kpLnext[:,0:1],kpLnext[:,1:2], c='g' ,label = 'kpL_T2')
#    plt.scatter(kpRnext[:,0:1],kpRnext[:,1:2], c='black' ,label = 'kpR_T2')
#    plt.xlim((0,2464))
#    plt.ylim((2056,0))
#    plt.xlabel("$x-direction(pixel)$")
#    plt.ylabel("$y-direction(pixel)$")
#    plt.legend()
####    
    

    #triangulate must be of the form 2 x N
    
    
#    x3dprev = (functions_3D.triangulate(kpLprev.T, kpRprev.T, Proj_1,Proj_2)).T
#    x3dnext = (functions_3D.triangulate(kpLnext.T, kpRnext.T, Proj_1,Proj_2)).T
#    print(x3dprev,x3dnext)
#    x3dprev = (functions_3D.triangulate(kpL_t1.T, kpR_t1.T, Proj_1,Proj_2)).T
#    x3dnext = (functions_3D.triangulate(kpL_t2.T, kpR_t2.T, Proj_1,Proj_2)).T
    print(x3dprev.shape,x3dnext.shape)
##    print(x3dprev)
#    x3dprev = np.expand_dims(x3dprev,axis =1 )
#    kpLprev =np.expand_dims(kpLprev,axis =1 )
#    kpLnext = np.expand_dims(kpLnext,axis =1 )
#    print(x3dprev.shape,kpLnext.shape)
#    
#    plt.figure(figsize=(10,10))
#    plt.scatter(corLprev_St[:,0:1],corLprev_St[:,1:2], c='r',label = 'kpL_T1')
#    plt.scatter(corRprev_St[:,0:1],corRprev_St[:,1:2], c='b',label = 'kpR_T1')
##    plt.scatter(kpLnext[:,0:1],kpLnext[:,1:2], c='g' ,label = 'kpL_T2')
##    plt.scatter(kpLnext[:,0:1],kpLnext[:,1:2], c='g' ,label = 'kpL_T2')
#    plt.xlim((0,2464))
#    plt.ylim((2056,0))
#    plt.xlabel("$x-direction(pixel)$")
#    plt.ylabel("$y-direction(pixel)$")
#    plt.legend()
    
    
    #Triangulate points
#    x3dprev = (functions_3D.triangulate(corLprev_St.T, corRprev_St.T, Proj_1,Proj_2)).T
#    x3dnext = (functions_3D.triangulate(corLnext.T, corRnext.T, Proj_1,Proj_2)).T
#    print(x3dprev.shape,x3dnext.shape)
#    print(x3dprev.shape)
#    x3dprev = x3dprev[:,:3]
#    x3dnext = x3dnext[:,:3]
#    
#    if len(x3dprev) >= len(x3dnext):
#        x3dprev = x3dprev[:len(x3dnext),:]
#    elif len(x3dprev) < len(x3dnext):
#        x3dnext = x3dnext[:len(x3dprev),:]
#    print(x3dprev,x3dnext)
#
#    M1_2, mask1_2 = cv2.findHomography(x3dprev, x3dnext, cv2.RANSAC,0.5)
#    mask1_2 = np.squeeze(mask1_2,axis =1)
##
#    x3dprev = x3dprev[mask1_2==1]
#    x3dnext = x3dnext[mask1_2==1]
#    a = np.array([[1,0,0]])
##    print(a)
#    b = np.array([[0,1,0]])
    x3dprev = np.array(x3dprev)
    x3dnext = np.array(x3dnext)   
#    x3dprev = np.array(x3dprev[:,:2])
#    x3dnext = np.array(x3dnext[:,:2])
#    
#    X = np.mean((x3dnext-x3dprev)[:,0])
#    Y = np.mean((x3dnext-x3dprev)[:,1])
#    Z = np.mean((x3dnext-x3dprev)[:,2])
#    np.asarray(X).ravel()
#    print(X,Y,Z)
#    print(x3dprev,x3dnext)
#    reg = rigid_registration(**{ np.asarray(X).ravel(), np.asarray(Y).ravel() })
    outlier = abs(x3dnext.shape[0]-x3dprev.shape[0])/x3dnext.shape[0]
#    print('outlier',outlier)
    start_t = time.time()
    t,R,T = functions_3D.register_rigid(x3dprev,x3dnext,0)
    end_t = time.time()
    CPD_T[idx-start_id,0], CPD_T[idx-start_id,1]= end_t-start_t,min(x3dprev.shape[0],x3dnext.shape[0])
    print(R,T,T[0])
    
    transx[idx-start_id]=T[0]
    transy[idx-start_id]=T[1]
    transz[idx-start_id]=T[2]

#    print(transx,transy,transz)
    cpd_plot = functions_3D.cpd_plot(x3dprev,x3dnext,t,idx)
    pointset[idx-start_id,0] = x3dprev.shape[0]
    pointset[idx-start_id,1] = x3dnext.shape[0]
##    a = rigid_registration()
#    R,T,iteration = functions_3D.icp(x3dprev,x3dnext)
#    print(R,T)
    
#    Delta_distance = np.sqrt(T[0,]**2+T[1,]**2)
#    print("Delta distance ")
    if math.isnan(T[0]) ==True or math.isnan(T[1]) ==True or math.isnan(T[2])==True:
        T = np.array([transx[idx-start_id-1],transy[idx-start_id-1],transz[idx-start_id-1]])
        R = np.eye(3)
        transx[idx-start_id]=T[0]
        transy[idx-start_id]=T[1]
        transz[idx-start_id]=T[2]
        ERROR[idx-start_id] = 1
        
    
    #set error threshold deltax= 0.1 deltay= 0.5 deltaz = 0.1
    if abs(T[0,]) >= 0.15 or abs(T[1,]) >= 0.5 or abs(T[1,]) <= 0.02 or abs(T[2,]) >= 0.1:
        T = np.array([transx[idx-start_id-1],transy[idx-start_id-1],transz[idx-start_id-1]])
        R = np.eye(3)
        transx[idx-start_id]=T[0]
        transy[idx-start_id]=T[1]
        transz[idx-start_id]=T[2]
        ERROR[idx-start_id] = 1
        
#    print(T)
#    print(R,T)
#    print(R,T.T)
#    transform = np.array([[X,Y]])
    
#    
    transform = functions_3D.makeTransform(R,T)
#    tvec1 = R.T.dot(T)
    print('R',R)
    ROT[idx-start_id] = math.asin(R[0,1])
    orientation = math.radians(-truePose[start_id,5])+sum(ROT[:idx-start_id])
    
    HEADING[idx-start_id] = orientation
    thetavo = np.ones((3,1))
    thetavo[0],thetavo[1],thetavo[2,] = math.radians(0),math.radians(0),orientation
    
    R = functions_3D.RotationMatrix(thetavo)
    tvec1 = R.T.dot(T)
    
    
    
    #Conpensate rotation matrix in camera frame
    theta = np.ones((3,1))
    theta[0], theta[1], theta[2] = math.radians(truePose[idx,4]),math.radians(-truePose[idx,3]),math.radians(-truePose[idx,5])
#    print('theta',theta)
    R_sensor = functions_3D.RotationMatrix(theta)
#    print(R_sensor)
#    distance = np.sqrt(tvec[0,]**2+tvec[1,]**2)
    tvec = R_sensor.T.dot(T)
#    print(tvec)
    distance = np.sqrt(tvec[0]**2+tvec[1]**2)
    distance_vo = np.sqrt(tvec1[0]**2+tvec1[1]**2)
    
    Time = truePose[idx,7]
#    print(tvec,tvec.shape)
    if idx ==start_id:
        pos=np.array([[0,0,0]])
#        pos = np.array([[truePose[start_id,1],truePose[start_id,0],truePose[start_id,6]]]) #EASTTING ,NORTHING, ALTITUDE.
        
        new_T = pos+tvec.T
        new_pos = pos+tvec1.T
        TVEC = np.concatenate((pos,new_T),axis =0)
        pos = np.concatenate((pos,new_pos),axis =0)
        Distrav = np.array([[0]])
        new_Distrav = Distrav+distance
        Distrav = np.concatenate((Distrav,new_Distrav),axis =0)
        
        Distrav_vo = np.array([[0]])
        new_Distrav_vo = Distrav_vo+distance_vo
        Distrav_vo = np.concatenate((Distrav_vo,new_Distrav_vo),axis =0)
        
        #Judge the direction in GT
#        if (truePose[idx,1]-truePose[idx+1,1]) >0:
        
        #Measure (R,t joint)
        pos_x = TVEC[idx-start_id,0]+truePose[start_id,1]
        pos_y = -TVEC[idx-start_id,1]+truePose[start_id,0]
        pos_z = TVEC[idx-start_id,2]+truePose[start_id,6]
        curError_x = abs(pos_x -truePose[idx,1])
        curError_y = abs(pos_y -truePose[idx,0])
        curError_z = abs(pos_z -truePose[idx,6])
        curError = np.sqrt(curError_x**2 + curError_y**2)  
        
        pos_x_vo = -pos[idx-start_id,0]+truePose[start_id,1]
        pos_y_vo = pos[idx-start_id,1]+truePose[start_id,0]
        pos_z_vo = pos[idx-start_id,2]+truePose[start_id,6]
        curError_x_vo = abs(pos_x_vo -truePose[idx,1])
        curError_y_vo = abs(pos_y_vo -truePose[idx,0])
        curError_z_vo = abs(pos_z_vo -truePose[idx,6])
        curError_vo = np.sqrt(curError_x_vo**2 + curError_y_vo**2)
        
        Sum = np.array([[ pos_x, pos_y, pos_z, curError_x, curError_y, curError_z, curError, idx, T[0,], T[1,], T[2,], -sum(transx[:idx-start_id])+truePose[start_id,1], sum(transy[:idx-start_id])+truePose[start_id,0], Distrav[idx-start_id], min(x3dnext.shape[0],x3dprev.shape[0]),Time]])
        
        Sum_vo = np.array([[ pos_x_vo, pos_y_vo, pos_z_vo, curError_x_vo, curError_y_vo, curError_z_vo, curError_vo, idx, Distrav_vo[idx-start_id], min(x3dnext.shape[0],x3dprev.shape[0]),Time]])
        
        STD = np.array([[np.mean(Sum[:,3]),np.mean(Sum[:,4]),np.mean(Sum[:,5]),np.mean(Sum[:,6]),np.std(Sum[:,3]),np.std(Sum[:,5]),np.std(Sum[:,5]),np.std(Sum[:,6]),altitude,altitude_std]])
        
        STD_vo = np.array([[np.mean(Sum_vo[:,3]),np.mean(Sum_vo[:,4]),np.mean(Sum_vo[:,5]),np.mean(Sum_vo[:,6]),np.std(Sum_vo[:,3]),np.std(Sum_vo[:,5]),np.std(Sum_vo[:,5]),np.std(Sum_vo[:,6]),altitude,altitude_std]])
        
        prev_pos = new_pos
        prev_T = new_T
        prev_Distrav = new_Distrav
        prev_Distrav_vo = new_Distrav_vo
        

    else:
        new_pos = prev_pos+tvec1.T
        pos = np.concatenate((pos,new_pos),axis =0)
        
        new_T = prev_T+tvec.T
        TVEC = np.concatenate((TVEC,new_T),axis =0)
        
        new_Distrav  = prev_Distrav+distance
        Distrav = np.concatenate((Distrav,new_Distrav),axis =0)
        
        new_Distrav_vo = prev_Distrav_vo+distance_vo
        Distrav_vo = np.concatenate((Distrav_vo,new_Distrav_vo),axis =0)
        
        

        pos_x = TVEC[idx-start_id,0]+truePose[start_id,1]
        pos_y = -TVEC[idx-start_id,1]+truePose[start_id,0]
        pos_z = TVEC[idx-start_id,2]+truePose[start_id,6]
        curError_x = abs(pos_x -truePose[idx,1])
        curError_y = abs(pos_y -truePose[idx,0])
        curError_z = abs(pos_z -truePose[idx,6])
        curError = np.sqrt(curError_x**2 + curError_y**2)
        
        pos_x_vo = -pos[idx-start_id,0]+truePose[start_id,1]
        pos_y_vo = pos[idx-start_id,1]+truePose[start_id,0]
        pos_z_vo = pos[idx-start_id,2]+truePose[start_id,6]
        curError_x_vo = abs(pos_x_vo -truePose[idx,1])
        curError_y_vo = abs(pos_y_vo -truePose[idx,0])
        curError_z_vo = abs(pos_z_vo -truePose[idx,6])
        curError_vo = np.sqrt(curError_x_vo**2 + curError_y_vo**2) 
        
        New_Sum = np.array([[ pos_x, pos_y, pos_z, curError_x, curError_y, curError_z, curError, idx, T[0,], T[1,], T[2,], -sum(transx[:idx-start_id])+truePose[start_id,1], sum(transy[:idx-start_id])+truePose[start_id,0] ,Distrav[idx-start_id], min(x3dnext.shape[0],x3dprev.shape[0]),Time]])
        Sum = np.concatenate((Sum,New_Sum),axis = 0)
        
        New_Sum_vo = np.array([[ pos_x_vo, pos_y_vo, pos_z_vo, curError_x_vo, curError_y_vo, curError_z_vo, curError_vo, idx, Distrav_vo[idx-start_id],min(x3dnext.shape[0],x3dprev.shape[0]),Time]])
        Sum_vo = np.concatenate((Sum_vo,New_Sum_vo),axis = 0)
        
        New_STD = np.array([[np.mean(Sum[:,3]),np.mean(Sum[:,4]),np.mean(Sum[:,5]),np.mean(Sum[:,6]),np.std(Sum[:,3]),np.std(Sum[:,5]),np.std(Sum[:,5]),np.std(Sum[:,6]),altitude,altitude_std]])
        STD = np.concatenate((STD,New_STD),axis = 0)
        
        New_STD_vo = np.array([[np.mean(Sum_vo[:,3]),np.mean(Sum_vo[:,4]),np.mean(Sum_vo[:,5]),np.mean(Sum_vo[:,6]),np.std(Sum_vo[:,3]),np.std(Sum_vo[:,5]),np.std(Sum_vo[:,5]),np.std(Sum_vo[:,6]),altitude,altitude_std]])
        STD_vo = np.concatenate((STD_vo,New_STD_vo),axis = 0)
        
        prev_pos = new_pos
        prev_T = new_T
        prev_Distrav = new_Distrav
        prev_Distrav_vo = new_Distrav_vo
##        
#    print('vo',pos_x,pos_y)
#    print('GT',truePose[idx,1],truePose[idx,0])
#    print('curError',curError)
#    print(Sum[:,12])
#    print(Sum[:,13])
#    print(TVEC)
#    print(pos)
#    print(Sum)
#    
    idx += 1
#    print(CPD_T)
#    print(T)
#    print(pos)
#    print(Sum[:,12])
#    print(Sum[:,13])
#    print('vo',pos_x,pos_y)
#    print('GT',truePose[idx-start_id,1],truePose[idx-start_id,0])
#    print('curError',curError)
    
#    print(matchKeptL_St,matchKeptR_St)
#    img_matches = np.empty((max(imLprev.shape[0], imRprev.shape[0]), imLprev.shape[1]+imRprev.shape[1], 3), dtype=np.uint8)
#    img3 = cv2.drawMatches(imLprev, kpLprev, imRprev, kpRprev, good1, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#    print((x3dprev),(pErrPrev),(matchKept_3d))
    
#    Test matches
#    cv2.namedWindow('Good Matches', 0)
#    cv2.imshow('Good Matches', img3)
#    cv2.waitKey()
    
#    kpLnext,descLnext = functions_3D.getFeatures_ORB(imLnext)
plt.figure(figsize=(8,8))
#plt.title('Trajectory',fontsize = 15)
plt.plot(Sum[:,0],Sum[:,1],c='g',linewidth=2,label = r'SVO + Sensor')
#plt.plot(-TVEC[:,0]+truePose[start_id,1],TVEC[:,1]+truePose[start_id,0],c='c',linewidth=2,label = r'SVO ($[R|T]$ measured dot)')
plt.plot(Sum[:,11],Sum[:,12],c='b',linewidth=2,label = r'SVO (Only $T$ measured)')
plt.plot(truePose[start_id:idx,1],truePose[start_id:idx,0],c='r',linewidth=2,label = r'Ground Truth')
plt.axis("equal")
plt.xlabel("Eastings $(m)$",fontsize = 15)
plt.ylabel("Northings $(m)$",fontsize = 15)
plt.legend(fontsize = 15)
plt.savefig('ONLYT.png', dpi =200, bbox_inches = 'tight')


plt.figure(figsize=(8,8))
#plt.title('Trajectory (only T)',fontsize = 15)
plt.plot(Sum[:,11],Sum[:,12],label = 'SVO')
plt.plot(truePose[start_id:idx,1],truePose[start_id:idx,0],c='r',label = 'Ground Truth')
plt.axis("equal")
plt.xlabel("Eastings $(m)$",fontsize = 15)
plt.ylabel("Northings $(m)$",fontsize = 15)
plt.legend(fontsize = 15)

plt.figure(figsize=(8,6))
plt.scatter(Sum[:,14],Sum[:,8],label = '$V_x$')
plt.scatter(Sum[:,14],Sum[:,9],c='r',label = '$V_y$')
#plt.scatter(Sum[:,14],Sum[:,10],c='g',label = '$V_z$')
plt.legend(fontsize = 15)
plt.xlabel("Inlier Feature",fontsize = 15)
plt.ylabel("Velocity (m/s)",fontsize = 15)
plt.savefig('VelocityFeature.png', dpi =200, bbox_inches = 'tight')

#print(Sum[:,12])
#print(Sum[:,13])
#
##
plt.figure(figsize=(8,4))
#plt.title('Error',fontsize = 15)
plt.plot(Sum[:,13],Sum[:,3],c='b',linewidth=2,label = 'x-Error')
plt.plot(Sum[:,13],Sum[:,4],c='r',linewidth=2,label = 'y-Error')
plt.plot(Sum[:,13],Sum[:,6],c='g',linewidth=2,label = 'Total Error')
plt.ylim(0,10)
plt.xlabel("Distance Traveled $(m)$",fontsize = 15)
plt.ylabel("Error $(m)$",fontsize = 15)
plt.legend(fontsize = 15)
plt.savefig('ErrorDistance.png', dpi =200, bbox_inches = 'tight')

plt.figure(figsize=(8,4))
#plt.title('Error',fontsize = 15)
plt.plot(Sum[:,13],STD[:,4],c='b',linewidth=2,label = 'x-Error')
plt.plot(Sum[:,13],STD[:,5],c='r',linewidth=2,label = 'y-Error')
plt.plot(Sum[:,13],STD[:,7],c='g',linewidth=2,label = 'Total Error')
plt.ylim(0,2)
plt.xlabel("Distance Traveled $(m)$",fontsize = 15)
plt.ylabel("Error standard deviation $(m)$",fontsize = 15)
plt.legend(fontsize = 15)
plt.savefig('ErrorSTD.png', dpi =200, bbox_inches = 'tight')

plt.figure(figsize=(8,4))
#plt.title('Error',fontsize = 15)
plt.plot(Sum[:,13],STD[:,0],c='b',linewidth=2,label = 'x-Error')
plt.plot(Sum[:,13],STD[:,1],c='r',linewidth=2,label = 'y-Error')
plt.plot(Sum[:,13],STD[:,3],c='g',linewidth=2,label = 'Total Error')
plt.ylim(0,10)
plt.xlabel("Distance Traveled $(m)$",fontsize = 15)
plt.ylabel("Error mean $(m)$",fontsize = 15)
plt.legend(fontsize = 15)
plt.savefig('ErrorMEAN.png', dpi =200, bbox_inches = 'tight')

plt.figure(figsize=(8,8))
#plt.title('Trajectory',fontsize = 15)
plt.plot(Sum[:,0],Sum[:,1],c='g',linewidth=2,label = r'SVO + Sensor')
plt.plot(Sum_vo[:,0],Sum_vo[:,1],c='b',linewidth=2,label = r'SVO')
plt.plot(truePose[start_id:idx,1],truePose[start_id:idx,0],c='r',linewidth=2,label = r'Ground Truth')
plt.axis("equal")
plt.xlabel("Eastings $(m)$",fontsize = 15)
plt.ylabel("Northings $(m)$",fontsize = 15)
plt.legend(loc='upper right',fontsize = 15)
plt.savefig('Trajectory.png', dpi =200, bbox_inches = 'tight')

plt.figure()
plt.plot(Sum[:,15]-truePose[start_id,7],STD[:,8],c='g',linewidth=2,label = r'SVO')
plt.plot(truePose[start_id:idx,7]-truePose[start_id,7],truePose[start_id:idx,6],c='r',linewidth=2,label = r'Ground Truth')
plt.xlabel("Time $(s)$",fontsize = 15)
plt.ylabel("Altitude $(m)$",fontsize = 15)
plt.ylim(0,3)
plt.legend(fontsize = 15)
plt.savefig('Altitude.png', dpi =200, bbox_inches = 'tight')

#x = np.std(Sum[:,3])
#y = np.std(Sum[:,4])
#err = np.std(Sum[:,6])
#print(x,y,err)

with open('svo_sequence1.csv', mode='w',newline='') as svo_file:
    svo_writer = csv.writer(svo_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    svo_writer.writerow(['Northings(m)', 'Eastings(m)','Altitude(m)','Error_x(m)','Error_y(m)','Error_z(m)','Total_Error(m)','Mean_Error(m)', 'Error_std(m)','Distance_traveled (m)','Timestamp','Error beyond threshold'])
    for i in range(0,Sum.shape[0]):
        northing = Sum[i,1]
        easting = Sum[i,0]
        altitude = Sum[i,2]
        err_x = Sum[i,3]
        err_y = Sum[i,4]
        err_z= Sum[i,5]
        err= Sum[i,6]
        err_mean = STD[i,3] 
        err_std = STD[i,7] 
        distance = Sum[i,13]
        t = Sum[i,15]
        err_b = ERROR[i,0]
        svo_writer.writerow([northing, easting,altitude,err_x, err_y, err_z, err,err_mean,err_std, distance, t,err_b])
        
        

with open('onlysvo_sequence1.csv', mode='w',newline='') as svo_file:
    svo_writer = csv.writer(svo_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    svo_writer.writerow(['Northings(m)', 'Eastings(m)','Altitude(m)','Error_x(m)','Error_y(m)','Error_z(m)','Total_Error(m)','Mean_Error(m)', 'Error_std(m)','Distance_traveled(m)','Timestamp','Error beyond threshold','heading(rad)'])
    for i in range(0,Sum.shape[0]):
        northing = Sum_vo[i,1]
        easting = Sum_vo[i,0]
        altitude = Sum_vo[i,2]
        err_x = Sum_vo[i,3]
        err_y = Sum_vo[i,4]
        err_z= Sum_vo[i,5]
        err= Sum_vo[i,6]
        err_mean = STD_vo[i,3] 
        err_std = STD_vo[i,7] 
        distance = Sum_vo[i,8]
        t = Sum[i,15]
        err_b = ERROR[i,0]
        heading = HEADING[i,0]
        svo_writer.writerow([northing, easting,altitude,err_x, err_y, err_z, err,err_mean,err_std, distance, t,err_b,heading])
        
with open('TIME.csv', mode='w',newline='') as svo_file:
    svo_writer = csv.writer(svo_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    svo_writer.writerow(['T','feature'])
    for i in range(0,Sum.shape[0]):
        time = CPD_T[i,0]
        feature = CPD_T[i,1]
        svo_writer.writerow([time,feature])
        
        
plt.figure()
plt.scatter(CPD_T[:,0],CPD_T[:,1])
##Sum = (Sum[i,1]).ravel()
#print(Sum)


##
#plt.figure(figsize=(8,8))
#plt.title('Number of Features',fontsize = 20)
#plt.plot(Sum[:,5],Sum[:,6],c='g',label = 'Feature number')
#
#plt.xlabel("Image Number",fontsize = 20)
#plt.ylabel("Tracked Features",fontsize = 20)
#plt.legend(fontsize = 20)


