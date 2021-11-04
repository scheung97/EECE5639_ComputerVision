#!/usr/bin/python3.7

import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy
from scipy import signal as sig

hallway_path = "./DanaHallWay1"
office_path = "./DanaOffice"


# !pip install opencv-python==3.4.2.17
# !pip install opencv-contrib-python==3.4.2.17

def image_gradients(img): 
    Ex = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    Ey = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    return Ex, Ey

def find_corner_coords(local_max): 
    #get corner locations
    corners = np.where(local_max>0)
    corners = np.array(corners).T
    return corners

def harrisCorner(img):
    """
    https://stackoverflow.com/questions/22974432/using-the-output-of-harris-corner-detector-to-detect-corners-a-new-issue-arises
    using Yung Yung's comment
    """
    window_size = 3 #for gaussian window
    std = 1
    k = 0.05
    
    #Image gradients using sobel filter:
    Ex, Ey = image_gradients(img)

    #gaussian window for smoothing
    w_mask = sig.windows.gaussian(window_size**2,std, sym=True) 
    w_mask = w_mask.reshape((window_size,window_size))
   
    #Compute products of derivatives :
    E2X = Ex**2
    E2Y = Ey**2
    Exy = Ex*Ey


    #Sums of products, using 9x9 gaussian matrix w/ std=1
    S2X = sig.convolve2d(E2X, w_mask) #top-left of C matrix
    S2Y = sig.convolve2d(E2Y, w_mask) #bottom-right of C matrix
    Sxy = sig.convolve2d(Exy, w_mask) #diagonals of C matrix

    #Calculate corner Response: 
    det = (S2X*S2Y) - (Sxy*Sxy)
    trace = (S2X + S2Y)
    
    R = det - k*((trace)**2)  
    
    #NMS
    _, R = cv2.threshold(R, 0.01*R.max(), 255, 0)

    pad_img = np.pad(R, ((1,1), (1,1)))
    local_max = np.zeros_like(pad_img) 

    for i in range(1, pad_img.shape[0]-1): 
        for j in range(1, pad_img.shape[1]-1): 
            wind = pad_img[i-1:i+1, j-1:j+1]
            if pad_img[i,j] == np.amax(wind): 
                local_max[i,j] = pad_img[i,j]

    #remove padding 
    corners = local_max[2:-2,2:-2] 
    
    return corners, (corners+img)

def get_window(img,h,w,n): 
    #gets nxn window of image centered at corner coords 
    #used for NCC calculation 
    pad_img = np.pad(img, ((n,n), (n,n)))
    window = pad_img[h+n : h+2*n, w+n : w+2*n]
    return window 

def match_corners(img1, img2, corners1, corners2):
    #matches corners between two images
    matches = np.zeros((np.sum(corners1>0), 5))
    corners1 = find_corner_coords(corners1)
    corners2 = find_corner_coords(corners2)
    for i in range(len(corners1)): 
        h1, w1 = corners1[i]
        match = np.array([h1,w1,0,0,0])
        window1 = get_window(img1,h1,w1,5)
        for j in range(len(corners2)):
            h2, w2 = corners2[j]
            window2 = get_window(img2,h2,w2,5)
            
            #NCC
            ncc = cv2.matchTemplate(window1.astype(np.float32),
                                    window2.astype(np.float32), 
                                    cv2.TM_CCORR_NORMED)
            if ncc >= match[4]: 
                match = [h1, w1, h2, w2, ncc]
        matches[i,:] = match 
    return matches 

def plot_correspondences(img1, img2, matches): 
    #draw lines between matching corners 
    h_offset= img1.shape[0]
    w_offset= img2.shape[1]
    x = np.copy(matches[:,0:4:2])
    y = np.copy(matches[:, 1:4:2])
    y[:,1] = y[:,1] + w_offset

    matched_imgs = np.hstack([img1,img2])
    plt.imshow(matched_imgs, cmap='gray')
    for i in range(0,x.shape[0]): 
        if matches[i,-1] >=0.99: 
            plt.plot(y[i,:], x[i,:])
    plt.show()
    return 0

def RANSAC(img1, img2): 
    #initialize sift 
    sift = cv2.xfeatures2d.SIFT_create()

    #find keypoints
    img1 = cv2.normalize(img1, None, 0,255, cv2.NORM_MINMAX).astype('uint8')
    img2 = cv2.normalize(img2, None, 0,255, cv2.NORM_MINMAX).astype('uint8')

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algo = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches2 = flann.knnMatch(des1, des2, k=2)

    #store all good matches: 
    good = []
    for m,n in matches2: 
        if m.dist < 0.7*n.distance: 
            good.append(m) 

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,matchesMask = matches_mask, flags = 2)#**draw_params)

    plt.imshow(img3, 'gray')
    plt.show()
    return M

def warp(img1, img2, H): 
    end_h = img1.shape[0]
    end_w = img1.shape[1] + img2.shape[1]
    
    warp_img1 = cv2.warpPerspective(img1, H, (end_w, end_h)) #warps img1 to img2 plane 

    # warp_img[0:img2.shape[0], 0:img2.shape[1]] = img2
    return (end_w, end_h), warp_img1

def blend(img1, img2): 
    alpha = 0.5
    beta = 1-alpha 
    blended = cv2.addWeighted(img1, alpha, img2, beta, 0)
    return blended 
    
def main(): 

 #1) read in 2 imgs (if img size is too large --> reduce, and document scale factor used)
    #Dana Office images: 
    office_files = [filename for filename in sorted(os.listdir(office_path)) if os.path.isfile(os.path.join(office_path, filename))]

    office_images = []    
    for i in range(0, len(office_files)): 
        img = cv2.imread(os.path.join(office_path, office_files[i]))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        office_images.append(gray_img)

    office_images = np.array(office_images)
    # print(office_images[1])

    #Dana Hallway images: 
    hallway_files = [filename for filename in sorted(os.listdir(hallway_path)) if os.path.isfile(os.path.join(hallway_path, filename))]

    hallway_imgs = []    
    for i in range(0, len(hallway_files)): 
        img = cv2.imread(os.path.join(hallway_path, hallway_files[i]))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        hallway_imgs.append(gray_img)

    hallway_files = np.array(hallway_imgs)

 #2) apply harris corner to both: compute Harris R function over image + do non-max suppression to get a sparse set of corner features
    #compute harris R function: 
    corners1, mask1 = harrisCorner(hallway_files[0])
    corners2, mask2 = harrisCorner(hallway_files[1])

 #3) Find correspondences btwn 2 imgs: Choose potential corner matches by finding pair of corners such that they have the highest NCC values (threshold for large NCC scores?)
    correspondences = match_corners(mask1, mask2, corners1, corners2)
    plot_correspondences(mask1, mask2, correspondences)


 #4) Correspondences --> estimate homography (Use RANSAC to help reduce outliers/errors)
        #breakdown of steps found in PDF
    M = RANSAC(mask1, mask2)
    
 #5) Warp image onto the other + blend overlapping pixels
        #breakdown of steps found in PDF
    (final_x, final_y), warped_mask1 = warp(mask1, mask2, M)
    blended_img = blend(warped_mask1, mask2)
    mosaic = cv2.bitwise_or(mask2,blended_img)
    # cv2.imshow(mosaic)
        
if __name__ == "__main__":
    main()



"""
extra credit info: 
https://learnopencv.com/homography-examples-using-opencv-python-c/  
"""