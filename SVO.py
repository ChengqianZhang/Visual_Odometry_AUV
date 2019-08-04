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


start_id = 84
end_id = 85
#Get the ground truth of the vehicle
maxError = 0
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

#Get camera parameters and calculate projection matirxs
camera_parameter_file_path_L = load_data_mission.get('params_path_LC', None)
camera_parameter_file_path_R = load_data_mission.get('params_path_RC', None)

Cammat_1,Cammat_2,Proj_1,Proj_2,dist_1,dist_2 = PnP_func.calc_projection(camera_parameter_file_path_L,camera_parameter_file_path_R)

print(Cammat_1,Cammat_2,Proj_1,Proj_2)
#Process all the images

size =[600,600]


traj = np.zeros((600, 600, 3), dtype=np.uint8)
truePose = PnP_func.getTruePose()

idx=start_id
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
    kpLprev,descLprev = PnP_func.bucket(imLprev,size)
    kpRprev,descRprev = PnP_func.bucket(imRprev,size)
    
    #get key points and descriputor orb
#    kpLprev,descLprev = PnP_func.getFeatures_ORB(imLprev)
#    kpRprev,descRprev = PnP_func.getFeatures_ORB(imRprev)
    print(descLprev.shape,descRprev.shape)
#    print(kpLprev)
#    print (len(kpLprev))
    ##correspondences
    corLprev_St, corRprev_St,matchKeptL_St,matchKeptR_St,good1 = PnP_func.getCorres(descLprev,descRprev,kpLprev,kpRprev,1)
#    print(matchKeptL_St,matchKeptR_St)
    img_matches = np.empty((max(imLprev.shape[0], imRprev.shape[0]), imLprev.shape[1]+imRprev.shape[1], 3), dtype=np.uint8)
    img3 = cv2.drawMatches(imLprev, kpLprev, imRprev, kpRprev, good1, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#    print((x3dprev),(pErrPrev),(matchKept_3d))
    
#    Test matches
    cv2.namedWindow('Good Matches', 0)
    cv2.imshow('Good Matches', img3)
    cv2.waitKey()
    
    #Detect the image at Tk_bucket
    kpLnext,descLnext = PnP_func.bucket(imLnext,size)
    
    #Detect the image at Tk and descriputor orb
#    kpLnext,descLnext = PnP_func.getFeatures_ORB(imLnext)
    
    corLprev_T,corLnext_T,matchKeptPrev_T,matchKeptNext_T,good2 = PnP_func.getCorres(descLprev,descLnext,kpLprev,kpLnext,1)
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
##    
    

    #triangulate must be of the form 2 x N
    x3dprev = (PnP_func.triangulate(kpLprev.T, kpRprev.T, Proj_1,Proj_2)).T
#    print(x3dprev)
    x3dprev = np.expand_dims(x3dprev,axis =2 )
    kpLprev =np.expand_dims(kpLprev,axis =2 )
    kpLnext = np.expand_dims(kpLnext,axis =2 )
    print(x3dprev.shape,kpLnext.shape)
#    print(x3dprev.shape,kpLnext.shape)
    
    #If the number of the 3d and corresponding 2d points are not enough, try the 2d-2d method for this image sequence
    if len(x3dprev) <= 5 and len(kpLnext) <= 5:
        # Get ground truth and scale
        truth_x, truth_y, truth_z, absolute_scale = PnP_func.getAbsoluteScale(filelist_LC, idx)

        E, mask = cv2.findEssentialMat(corLprev_T.T,corLnext_T.T,  Cammat_1, cv2.RANSAC,0.999,1.0)
        _, R, t, mask = cv2.recoverPose(E, corLprev_T.T,corLnext_T.T, Cammat_1)
        if absolute_scale > 0.1:  
            trans = absolute_scale*R.dot(t)
            rot = R.dot(R)
        print('Point not enough use 2D-2D',idx, 't',trans)
    else: 
        rot,trans=PnP_func.camPose(x3dprev, kpLnext, Cammat_1)
#    print(rot,trans)
        
        
    transform = PnP_func.makeTransform(rot,trans)
#    print(transform)
    
    if idx ==start_id:
        
        x3d_Saved=(x3dprev)
        pos = np.array([[0,0,0]])
        newpos = PnP_func.posUpdate(pos,transform)
        pos = np.concatenate((pos,newpos),axis =0)
        delta = newpos-pos
        prevpos = newpos
        

    else:
        x3d_Saved=np.concatenate((x3d_Saved,x3dprev),axis=0)
        newpos = PnP_func.posUpdate(prevpos,transform)
        pos = np.concatenate((pos,newpos),axis =0)
        delta = newpos-prevpos
        prevpos = newpos
#    print(newpos-prevpos)
    print(delta)
    print(pos)
    #draw images
    #Tarnsfrom the coordinate form vehicle model to initial
    
    
    x,y,z =  PnP_func.body_to_inertial(truePose[idx][3],truePose[idx][4],truePose[idx][5],delta[0,1],delta[0,0],-delta[0,2])
    print('x,y',x,y)
    if idx ==start_id:
        prev_x,prev_y =  truePose[start_id][1], truePose[start_id][0]
        new_x,new_y =y + prev_x, x + prev_y
        draw_x, draw_y = int(new_x), int(new_y)
        true_x, true_y = int(truePose[idx+1][1]), int(truePose[idx+1][0])
        print(new_x,new_y)
        print(truePose[idx+1][1], truePose[idx+1][0])

        curError = np.sqrt((new_x-truePose[idx+1][1])**2 + (new_y-truePose[idx+1][0])**2)
        curError_x = abs(new_x-truePose[idx+1][1])
        curError_y = abs(new_y-truePose[idx+1][0])
        position = np.array([[new_x,new_y,curError_x,curError_y,curError,len(kpLnext),idx]])
        #(tvec[1]-truePose[i][7])**2 
        prev_x = new_x
        prev_y = new_y
        print('Current Error: ', curError)
        print('Current Error_x: ', curError_x)
        print('Current Error_y: ', curError_y)
        if (curError > maxError):
            maxError = curError
    else:
        new_x,new_y =y + prev_x, x + prev_y
        draw_x, draw_y = int(new_x), int(new_y)
        true_x, true_y = int(truePose[idx+1][1]), int(truePose[idx+1][0])
        print(new_x,new_y)
        print(truePose[idx+1][1], truePose[idx+1][0])

        curError = np.sqrt((new_x-truePose[idx+1][1])**2 + (new_y-truePose[idx+1][0])**2)
        curError_x = abs(new_x-truePose[idx+1][1])
        curError_y = abs(new_y-truePose[idx+1][0])
        newposition = np.array([[new_x,new_y,curError_x,curError_y,curError,len(kpLnext),idx]])
        position = np.concatenate((position,newposition),axis = 0)#(tvec[1]-truePose[i][7])**2 
        prev_x = new_x
        prev_y = new_y
        print('Current Error: ', curError)
        print('Current Error_x: ', curError_x)
        print('Current Error_y: ', curError_y)
        if (curError > maxError):
            maxError = curError
#    if maxError > 5 
        
        

            # print([truePose[i][3], truePose[i][7], truePose[i][11]])

    text = "Coordinates: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(newpos[0,0]), float(newpos[0,1]), float(newpos[0,2]))
    cv2.circle(traj, (draw_x, draw_y) ,1, (0,0,255), 2)
    cv2.circle(traj, (true_x, true_y) ,1, (255,0,0), 2)
    cv2.rectangle(traj, (10, 30), (550, 50), (0,0,0), cv2.FILLED);
    cv2.putText(traj, text, (10,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    cv2.imshow( "Trajectory", traj )
    k = cv2.waitKey(1) & 0xFF
    if k == 27: break

    #cv2.waitKey(0)
    print('Maximum Error: ', maxError)
    print(position)
    cv2.imwrite('map2.png', traj)
    idx += 1

plt.figure(figsize=(8,8))
plt.title('3D-2D feature tracked from ORB_detector',fontsize = 20)
plt.plot(position[:,6],position[:,5])
plt.xlabel("$Image index$",fontsize = 20)
plt.ylabel("$Number of feature$",fontsize = 20)
plt.legend(fontsize = 20)


plt.figure(figsize=(8,8))
plt.title('3D-2D Error from ORB_detector',fontsize = 20)
plt.plot(position[:,6],position[:,3],label = 'Error (y)')
plt.plot(position[:,6],position[:,2],label = 'Error (x)')
plt.xlabel("$Image index$",fontsize = 20)
plt.ylabel("$Current Error$",fontsize = 20)
plt.legend(fontsize = 20)