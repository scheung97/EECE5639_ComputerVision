import cv2 
import numpy as np
import math




def main(): 
    #read in 2 imgs (if img size is too large --> reduce, and document scale factor used)
    
    #apply harris corner detector to both: compute Harris R function over image + do non-max suppression to get a sparse set of corner features

    #Find correspondences btwn 2 imgs: Choose potential corner matches by finding pair of corners such that they have the highest NCC values (threshold for large NCC scores?)

    #Correspondences --> estimate homography (Use RANSAC to help reduce outliers/errors)
        #breakdown of steps found in PDF

    #Warp image onto the other + blend overlapping pixels
        #breakdown of steps found in PDF
    
    


if __name__ == "__main__":
    main()



"""
extra credit info: 
https://learnopencv.com/homography-examples-using-opencv-python-c/
"""