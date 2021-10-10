#GOAL: detect moving object by looking at large gradients in the temporal evolution of pixael values
import cv2 
import numpy as np 
from scipy import signal
import matplotlib.pyplot as plt
import os

def process_frame(frame): 
    # grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    oneDimOperator = np.matrix([-1, 0, 1])

    filtImage = signal.convolve2d(frame, oneDimOperator)

    return filtImage
def main():    
    #file paths to images
    officePath = r'../Office/'
    officeGrayPath = r'../OfficeGray/'
    redChairPath = r'../RedChair'
    redChairGrayPath = r'../RedChairGray/'

    #variables used (update when changing image folders): 
    imagePath = redChairPath #officePath
    pathUsed = redChairGrayPath #officeGrayPath 

    #check if file
    if(os.path.exists(pathUsed) == False):
        print('--------------------------------------------------------')
        print('\tMaking gray image directory....')
        print('--------------------------------------------------------')
        os.mkdir(pathUsed)
    
    if(len(os.listdir(pathUsed)) == 0): 
        print('--------------------------------------------------------')
        print('\tCreating grayscale images....')
        print('--------------------------------------------------------')
        files = [f for f in os.listdir(imagePath) if os.path.isfile(os.path.join(imagePath,f))]
        imgs = np.empty(len(files), dtype=object)
        grayImgs = np.empty(len(imgs), dtype=object)
        
        for i in range(0, len(imgs)): 
            imgs[i] = cv2.imread(os.path.join(imagePath,files[i]))
            grayImgs[i] = cv2.cvtColor(imgs[i],cv2.COLOR_BGRA2GRAY)
            filename = "{}grayimg{}.jpg".format(pathUsed,i)
            cv2.imwrite(filename, grayImgs[i])



        # filtImg=process_frame(grayImgs[i]) #convoolve over 1 frame
        # cv2.imshow('deep', filtImg)
    


    #with enough frames available, apply 1-D diff operator to compute temporal derivative
    #threshold absolute values of derivatives to create 0 and 1 mask of moving objects
    #combine mask with original frame to display results

    """
    Variations: 
        - 0.5[-1,0,1] filter and 1D derivative of Gaussian (usr defined std of tsigma)
        - 2D spatial smoothing before temporal filter; 
            - try 3x3, 5x5 and 2D Gaussian filters (usr defined std of ssigma)
        - Vary threshold + design strat to select good threshold for each image 
            - hint: bkgrnd pixels have temporal gradients close to zero --> model these vals as Gaussian 0-mean noise              + estimate std of this noise
    """


       



if __name__ == "__main__": 
    main()
    
