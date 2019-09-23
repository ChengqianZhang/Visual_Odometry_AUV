# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 20:47:47 2019

@author: sheva
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import cv2
import math
import operator
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt





def cpd_p(x, y, sigma2, w, m, n, d):
    """
    E-step:Compute P in the EM optimization,which store the probobility of point n in x belongs the cluster m in y.
    Parameters
    ----------
    x : ndarray
        The static shape that y will be registered to. Expected array shape is [n_points_x, n_dims]
    y : ndarray
        The moving shape. Expected array shape is [n_points_y, n_dims]. Note that n_dims should be equal for x and y,
        but n_points does not need to match.
    sigma2 : float
        Gaussian distribution parameter.It will be calculated in M-step every loop.
    w : float
        Weight for the outlier suppression. Value is expected to be in range [0.0, 1.0].
    m : int
        x points' length. The reason that making it a parameter here is for avioding calculate it every time.
    n : int
        y points' length. The reason that making it a parameter here is for avioding calculate it every time.
    d : int
        Dataset's dimensions. Note that d should be equal for x and y.
    Returns
    -------
    p1 : ndarray
        The result of dot product of the matrix p and a column vector of all ones.
        Expected array shape is [n_points_y,1].
    pt1 : nadarray
        The result of dot product of the inverse matrix of p and a column vector of all ones. Expected array shape is
        [n_points_x,1].
    px : nadarray
        The result of dot product of the matrix p and matrix of dataset x.
    """
    # using numpy broadcasting to build a new matrix.
    g = x[:, np.newaxis, :]-y
    g = g*g
    g = np.sum(g, 2)
    g = np.exp(-1.0/(2*sigma2)*g)
    # g1 is the top part of the expression calculating p
    # temp2 is the bottom part of expresion calculating p
    g1 = np.sum(g, 1)
    temp2 = (g1 + (2*np.pi*sigma2)**(d/2)*w/(1-w)*(float(m)/n)).reshape([n, 1])
    p = (g/temp2).T
    p1 = (np.sum(p, 1)).reshape([m, 1])
    px = np.dot(p, x)
    pt1 = (np.sum(np.transpose(p), 1)).reshape([n, 1])
    return p1, pt1, px
def register_rigid(x, y, w, max_it=2000):
    """
    Registers Y to X using the Coherent Point Drift algorithm, in rigid fashion.
    Note: For affine transformation, t = scale*y*r'+1*t'(* is dot). r is orthogonal rotation matrix here.
    Parameters
    ----------
    x : ndarray
        The static shape that y will be registered to. Expected array shape is [n_points_x, n_dims]
    y : ndarray
        The moving shape. Expected array shape is [n_points_y, n_dims]. Note that n_dims should be equal for x and y,
        but n_points does not need to match.
    w : float
        Weight for the outlier suppression. Value is expected to be in range [0.0, 1.0].
    max_it : int
        Maximum number of iterations. The default value is 150.
    Returns
    -------
    t : ndarray
        The transformed version of y. Output shape is [n_points_y, n_dims].
    """
    # get dataset lengths and dimensions
    [n, d] = x.shape
    [m, d] = y.shape
    # t is the updated moving shape,we initialize it with y first.
    t = y
    # initialize sigma^2
    sigma2 = (m*np.trace(np.dot(np.transpose(x), x))+n*np.trace(np.dot(np.transpose(y), y)) -
              2*np.dot(sum(x), np.transpose(sum(y))))/(m*n*d)
    iter = 0
    while (iter < max_it) and (sigma2 > 10.e-8):
        [p1, pt1, px] = cpd_p(x, t, sigma2, w, m, n, d)
        # precompute
        Np = np.sum(pt1)
        mu_x = np.dot(np.transpose(x), pt1)/Np
        mu_y = np.dot(np.transpose(y), p1)/Np
        # solve for Rotation, scaling, translation and sigma^2
        a = np.dot(np.transpose(px), y)-Np*(np.dot(mu_x, np.transpose(mu_y)))
        [u, s, v] = np.linalg.svd(a)
        s = np.diag(s)
        c = np.eye(d)
        c[-1, -1] = np.linalg.det(np.dot(u, v))
        r = np.dot(u, np.dot(c, v))
        scale = np.trace(np.dot(s, c))/(sum(sum(y*y*np.matlib.repmat(p1, 1, d)))-Np *
                                        np.dot(np.transpose(mu_y), mu_y))
        sigma22 = np.abs(sum(sum(x*x*np.matlib.repmat(pt1, 1, d)))-Np *
                         np.dot(np.transpose(mu_x), mu_x)-scale*np.trace(np.dot(s, c)))/(Np*d)
        sigma2 = sigma22[0][0]
        # ts is translation
        ts = mu_x-np.dot(scale*r, mu_y)
        t = np.dot(scale*y, np.transpose(r))+np.matlib.repmat(np.transpose(ts), m, 1)
        iter = iter+1
    return t,r,ts

def cpd_plot(x, y, t,idx):
    """
    Plot the initial datasets and registration results.
    Parameters
    ----------
    x : ndarray
        The static shape that y will be registered to. Expected array shape is [n_points_x, n_dims]
    y : ndarray
        The moving shape. Expected array shape is [n_points_y, n_dims]. Note that n_dims should be equal for x and y,
        but n_points does not need to match.
    t : ndarray
        The transformed version of y. Output shape is [n_points_y, n_dims].
    """
    if len(x[0, :]) == 2:
        plt.figure(1)
        plt.plot(x[:, 0], x[:, 1], 'go',label='3D $T_k$')
        plt.plot(y[:, 0], y[:, 1], 'r+',label='3D $T_(k+1$)')
        plt.title("Before registration",fontsize = 12)
        plt.xlabel("$X (m)$",fontsize = 12)
        plt.ylabel("$Y (m)$",fontsize = 12)
        plt.legend(fontsize = 12)
        plt.tight_layout()
        plt.savefig('Before2D.png', dpi=200,bbox_inches = 'tight')
        plt.figure(2)
        plt.plot(x[:, 0], x[:, 1], 'go',label='3D $T_k$')
        plt.plot(t[:, 0], t[:, 1], 'r+',label='3D $T_(k+1$)')
        plt.title("After registration",fontsize = 12)
        plt.xlabel("$X (m)$",fontsize = 12)
        plt.ylabel("$Y (m)$",fontsize = 12)
        plt.legend(fontsize = 12)
        plt.tight_layout()
        plt.savefig('After2D.png', dpi=200,bbox_inches = 'tight')
        plt.show()
    elif len(x[0, :]) == 3:
        ax1 = Axes3D(plt.figure(1))
        ax1.plot(x[:, 0], x[:, 1], x[:, 2], 'go',label='3D $T_k$')
        ax1.plot(y[:, 0], y[:, 1], y[:, 2], 'r+',label='3D $T_(k+1$)')
        ax1.set_title("Before registration", fontdict=None, loc='center',fontsize = 12)
        ax1.set_xlabel("$X (m)$",fontsize = 12)
        ax1.set_ylabel("$Y (m)$",fontsize = 12)
        ax1.set_zlabel("$Z (m)$",fontsize = 12)
        plt.legend(fontsize = 12)
        plt.tight_layout(False)
        plt.savefig('Before3D-{:}.png'.format(idx), dpi=120,bbox_inches = 'tight')
        ax2 = Axes3D(plt.figure(2))
        ax2.plot(x[:, 0], x[:, 1], x[:, 2], 'go',label='3D $T_k$')
        ax2.plot(t[:, 0], t[:, 1], t[:, 2], 'r+',label='3D $T_(k+1$)')
        ax2.set_title("After registration", fontdict=None, loc='center',fontsize = 12)
        ax2.set_xlabel("$X (m)$",fontsize = 12)
        ax2.set_ylabel("$Y (m)$",fontsize = 12)
        ax2.set_zlabel("$Z (m)$",fontsize = 12)
        plt.legend(fontsize = 12)
        plt.tight_layout(False)
        plt.savefig('After3D-{:}.png'.format(idx), dpi=120,bbox_inches = 'tight')
        plt.show()


def KLT(img_1, img_2, p1):
    ##use KLT tracker
    lk_params = dict( winSize  = (21,21),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
    st = st.reshape(st.shape[0])
    ##find good one
    p1 = p1[st==1]
    p2 = p2[st==1]

    return p1,p2,st




def match_points(source, target, max_correspondence_search=None):
    if max_correspondence_search is None:
        if source.shape[0] <= target.shape[0]:
            max_correspondence_search = source.shape[0]
        else:
            max_correspondence_search = target.shape[0]
    
    # get the nearest neighbors up to some depth
    # the maximum depth gives the full distance matrix
    # this will guarantee a 1-to-1 correspondence between points by 
    # distance, however it could be very slow for large datasets
    nn = NearestNeighbors(n_neighbors=max_correspondence_search)
    nn.fit(target)
    distances, indicies = nn.kneighbors(source, return_distance=True)
    # this will give us a list of the row and column indicies (in the distance matrix)
    # of the distances in increasing order
    dist_argsort_row, dist_argsort_col = np.unravel_index(distances.ravel().argsort(), distances.shape)
    source_idxs = []
    target_idxs = []
    dists = []
    
    for dar, dac in zip(dist_argsort_row, dist_argsort_col):
        if dar not in source_idxs:
            tidx = indicies[dar, dac]
            if tidx not in target_idxs:
                source_idxs.append(dar)
                target_idxs.append(tidx)
                dists.append(distances[dar, dac])
                
    
    return np.array(dists), np.array(source_idxs), np.array(target_idxs)


def compute_transform(source, target, return_as_single_matrix=False):
    # basic equation we are trying to optimize
    # error = target - scale*Rot*source - translation
    # so we need to find the scale, rotation, and translation 
    # that minimizes the above equation
    
    # based on notes from
    # https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    
    # made more clear from http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    
    # in this implementation I assume that the matricies source and target are of 
    # the form n x m, where n is the number of points in each matrix
    # and m is the number of dimensions
    
    assert source.shape == target.shape
    n, m = source.shape
    
    # compute centroids
    
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    
    # this removes the translation component of the transform
    # we can estimate the translation by the centroids
    source_rel = source - source_centroid
    target_rel = target - target_centroid
    
    source_var = source_rel.var()
    
    # next we need to estimate the covariance matrix between
    # the source and target points (should be an mxm matrix)
    # in the literature this is often denoted as "H"
    
    H = target_rel.T.dot(source_rel) / (n*m)
    
    
    # now we can do SVD on H to get the rotation matrix
    
    U, D, V = np.linalg.svd(H)
    
    # rotation - first check the determinants of U and V
    u_det = np.linalg.det(U)
    v_det = np.linalg.det(V)
    
    S = np.eye(m)
    
    if u_det*v_det < 0.0:
        S[-1] = -1
    
    rot = V.T.dot(S).dot(U)
    
    # compute the scale
    scale = (np.trace(np.diag(D).dot(S)) / source_var)
    
    # compute the translation
    trans = target_centroid - source_centroid.dot(rot*scale)
    
    if return_as_single_matrix:
        T = np.eye(m+1)
        T[:m,:m] = scale * rot
        T[m,:m] = trans
        return T
    
    
    return rot, trans, scale


def icp(source, target, max_iter=100, tol=1e-6, d_tol=1e-10, max_correspondence_search=None):
    sn, sm = source.shape
    tn, tm = target.shape
    
    assert sm == tm, "Dimensionality of point sets must be equal"
    
    S = source.copy()
    T = target.copy()
    
    # initialize the scale, rot, and translation estimates
    # here we will the respective centroids and scales get an initial correspondence between
    # the two point sets
    # if we just take the raw distances,
    # all the points in the source may map to one target and vice versa
    Sc = ( (S-S.mean(axis=0)) / S.std(axis=0))
    Tc = ( (T-T.mean(axis=0)) / T.std(axis=0))

    d,s_idxs, t_idxs = match_points(Sc, Tc, max_correspondence_search=max_correspondence_search)

    rotation, _, _ = compute_transform( Sc[s_idxs, :], Tc[t_idxs, :] )
    scale = 1.0
    translation = T.mean(axis=0) - S.mean(axis=0).dot(scale * rotation)
    S = S.dot(scale*rotation) + translation

    prev_err = 1e6
    n_pt = 0
    for i in range(max_iter):
    
        # match the closest points based on some distance metric (using the sklearn NearestNeighbor object)
        d,s_idxs,t_idxs = match_points(S, T, max_correspondence_search=max_correspondence_search)

        # estimate the translation/rotation/scaling to match the source to the target
        rotation, translation, scale = compute_transform(S[s_idxs, :], T[t_idxs, :])

        # transform the source (i.e. update the positions of the source)
        S = S.dot(scale*rotation) + translation

        # repeat until convergence or max iterations
        err = np.mean(d)

        if np.abs(prev_err - err) <= tol:
            break
        
        prev_err = err
            
    rotation, translation, scale = compute_transform(source, S) # we know the exact correspondences here
    return rotation, np.array(translation), scale


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

def getTruePoseKITTI():
    file = 'D:/Southampton/Msc Project/TEST FOLDER/CameraTrajectory.txt'
    return np.genfromtxt(file, delimiter=' ',dtype=None)
def getFeatures_SIFT(img):
    sift = cv2.xfeatures2d.SIFT_create(
            nfeatures = 5000, 
            nOctaveLayers = 3,
            contrastThreshold =0.035, 
            edgeThreshold = 30)
    kp, desc = sift.detectAndCompute(img, None)
    return kp,desc
def getFeatures_SIFT_bucket(img, sizeWindowArray ):
    
    height, width= img.shape

    xstep = int(width/sizeWindowArray[1])
    ystep = int(height/sizeWindowArray[0])
    sift = cv2.xfeatures2d.SIFT_create(
            nfeatures = 1000, 
            nOctaveLayers = 3,
            contrastThreshold =0.02, 
            edgeThreshold = 30)
    
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
    detector = cv2.ORB_create(
            nfeatures=5000,
            scaleFactor=1.2,
            nlevels= 8,
            edgeThreshold=31,
            firstLevel= 0,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20)
#    detector = cv2.ORB_create()
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

    
    leftCorres = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    rightCorres = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
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

def camPose(x3d_T1,x3d_T2):
    
    T, distances, i = icp(x3d_T1,x3d_T2)  
#    tvec = rot.dot(tvec)      #tvec = -rot.T.dot(tvec)  #coordinate transformation, from camera to worl
    return T,i


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


def stereo_match_feature(left_img, right_img, patch_radius, keypoints, min_disp, max_disp):  
    # in case you want to find stereo match by yourself
    h, w = left_img.shape
    num_points = keypoints.shape[0]

    # Depth (or disparity) map
#    depth = np.zeros(left_img.shape, np.uint8)
    output = np.zeros(keypoints.shape, dtype='int')
    all_index = np.zeros((keypoints.shape[0],1), dtype='int').reshape(-1)

    r     = patch_radius
    # patch_size = 2*patch_radius + 1;
      
    for i in range(num_points):

        row, col = keypoints[i,0], keypoints[i,1]
        # print(row, col)
        best_offset = 0;
        best_score = float('inf');

        if (row-r < 0 or row + r >= h or col - r < 0 or col + r >= w): continue

        left_patch = left_img[(row-r):(row+r+1), (col-r):(col+r+1)] # left imag patch    

        all_index[i] = 1

        for offset in range(min_disp, max_disp+1):

              if (row-r) < 0 or row + r >= h or  (col-r-offset) < 0 or (col+r-offset) >= w: continue
        
              diff  = left_patch - right_img[(row-r):(row+r+1), (col-r-offset):(col+r-offset+1)]
              sum_s = np.sum(diff**2)
 
              if sum_s < best_score:
                  best_score = sum_s
                  best_offset = offset

        output[i,0], output[i,1] = row,col-best_offset

    return output, all_index

#COMPENSTAE THE ROTATION BY CHEAP SENSORS
def RotationMatrix(theta):
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
    R_z * R_y * R_x
 
    return R