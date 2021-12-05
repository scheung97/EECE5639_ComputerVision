import cv2  
import numpy as np
import os 
from matplotlib import pyplot as plt
from scipy import signal as sig

cast_path = r'./cast_imgs'
cones_path = r'./cones_imgs'

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




def main(): 
    #read in cast images: 
    cast_files = [filename for filename in sorted(os.listdir(cast_path)) if os.path.isfile(os.path.join(cast_path, filename))]
    cast_imgs = []    
    for i in range(0, len(cast_files)): 
        img = cv2.imread(os.path.join(cast_path, cast_files[i]))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        cast_imgs.append(gray_img)
    cast_imgs = np.array(cast_imgs)

    #read in cones images: 
    cones_files = [filename for filename in sorted(os.listdir(cones_path)) if os.path.isfile(os.path.join(cones_path, filename))]
    cones_imgs = []    
    for i in range(0, len(cones_files)): 
        img = cv2.imread(os.path.join(cones_path, cones_files[i]))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        cones_imgs.append(gray_img)
    cones_imgs = np.array(cones_imgs)

# 1.  Find interesting features and correspondences between the left and right images. You can
# use the CORNERS and NCC algorithms that you wrote/used for the second project or SIFT
# features and descriptors. Display your results in the same way you did for project 2 i.e. by
# connecting corresponding features with a line. Using lines of different colors for different
# points makes it easier to visualize the results.

    # #correspondences for cast imgs
    # cast_corners1, cast_mask1 = harrisCorner(cast_imgs[0])
    # cast_corners2, cast_mask2 = harrisCorner(cast_imgs[1])
    # cast_correspondences = match_corners(cast_mask1, cast_mask2, cast_corners1, cast_corners2)
    # plot_correspondences(cast_mask1, cast_mask2, cast_correspondences)
    #correspondences for cones imgs
    cones_corners1, cones_mask1 = harrisCorner(cones_imgs[0])
    cones_corners2, cones_mask2 = harrisCorner(cones_imgs[1])
    cones_correspondences = match_corners(cones_mask1, cones_mask2, cones_corners1, cones_corners2)
    plot_correspondences(cones_mask1, cones_mask2, cones_correspondences)


    #toggle for using specific image set: 
    correspondences = cones_correspondences 

# 2. Write a program to estimate the Fundamental Matrix for each pair using the correspondences
# above and RANSAC to eliminate outliers. Display the inlier correspondences in the same
# way as above.
    
    """ https://www.programcreek.com/python/example/89336/cv2.findFundamentalMat
        http://www.opencv.org.cn/opencvdoc/2.3.2/html/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=findfun
    """

    # print(cast_correspondences[:][2][0])
    pts1 = [] 
    pts2 = []

    for i in range(len(correspondences)):
        temp1 = [correspondences[:][i][0],correspondences[:][i][1]]
        temp2 = [correspondences[:][i][2],correspondences[:][i][3]]
        pts1.append(temp1)
        pts2.append(temp2)
    pts1 = np.array(pts1) 
    pts2 = np.array(pts2)
    # print(pts1)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1,0.99) #0.1 = threshold dist from pixel; 0.99 = confidence 
        #thresh_dist = max distance from a point to an epipolar line in pixels, beyond which the point is considered an outlier 
        #confidence = specify desired level of confidence(probability) that estimated matrix is correct
# 3. Compute a dense disparity map using the Fundamental matrix to help reduce the search
# space. The output should be three images, one image with the vertical disparity component,
# and another image with the horizontal disparity component, and a third image representing
# the disparity vector using color, where the direction of the vector is coded by hue, and the
# length of the vector is coded by saturation. For gray scale display, scale the disparity values
# so the lowest disparity is 0 and the highest disparity is 255.

    """https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html"""
    # Find epipolar lines corresponding to points in right image(2nd img) and draw its lines on left image) 
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2),2,F)
    lines1 = lines1.reshape(-1,3)

    # Find epipolar lines corresponding to points in left image(1st img) and draw its lines on right image) 
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2),1,F)
    lines2 = lines2.reshape(-1,3)

    """
    Math way to get disparity(?):
    line_l = F * pts1 #ax+by+c = 0 --> by+c (using purely horizontal lines)
    
    best_pts = search_lines(img1, line_l)
    disp = pts1-best_pts 
    
    """

    min_disp = 16
    num_disp = 64-min_disp

    stereo = cv2.StereoSGBM_create(minDisparity = min_disp, 
                                numDisparities = num_disp, blockSize = 1)
    disp = stereo.compute(cones_imgs[0], cones_imgs[1]).astype(np.float32) / 16.0
    plt.imshow(disp, 'gray') 
    plt.figure(2)
    plt.imshow(disp, 'heatmap')
    plt.show()
    return
if __name__ == "__main__": 
    main()