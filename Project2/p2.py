import cv2 
import numpy as np
import math
import os 
import scipy
from scipy import signal as sig

hallway_path = "./DanaHallWay1"
office_path = "./DanaOffice"

def edge_detection(img): 
    sobel_X= np.array([[-1,0, 1],[-2,0,2],[-1,0,-1]])
    sobel_Y= np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
    Ex = sig.convolve2d(img, sobel_X)
    Ey = sig.convolve2d(img, sobel_Y)
    return Ex, Ey

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
    # print(hallway_files.shape)

 #2) apply harris corner detector to both: compute Harris R function over image + do non-max suppression to get a sparse set of corner features
    Ex, Ey = edge_detection(hallway_files[0])
    print(Ex) 
    print('============================')
    print(Ey)
 #3) Find correspondences btwn 2 imgs: Choose potential corner matches by finding pair of corners such that they have the highest NCC values (threshold for large NCC scores?)

 #4) Correspondences --> estimate homography (Use RANSAC to help reduce outliers/errors)
        #breakdown of steps found in PDF

 #5) Warp image onto the other + blend overlapping pixels
        #breakdown of steps found in PDF
    
    


if __name__ == "__main__":
    main()



"""
extra credit info: 
https://learnopencv.com/homography-examples-using-opencv-python-c/
"""