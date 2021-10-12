#GOAL: detect moving object by looking at large gradients in the temporal evolution of pixael values
import cv2 
import numpy as np
from numpy.core.function_base import linspace
from numpy.lib.twodim_base import mask_indices 
from scipy import signal
import matplotlib.pyplot as plt
import os

def thresholding(frame, threshold): 
    th,dst=cv2.threshold(frame,threshold, 255,cv2.THRESH_BINARY)
    return dst



def main():    
    ##file paths to images
    officePath = r'../Office/'
    officeGrayPath = r'../OfficeGray/'
    redChairPath = r'../RedChair'
    redChairGrayPath = r'../RedChairGray/'

    ## variables used (update when changing image folders): 
    # imagePath = redChairPath 
    imagePath = officePath
    # pathUsed = redChairGrayPath 
    pathUsed = officeGrayPath 


    #check if file directiory exists: 
    if(os.path.exists(pathUsed) == False):
        print('--------------------------------------------------------')
        print('Making gray image directory....')
        print('--------------------------------------------------------')
        os.mkdir(pathUsed)
    
    #check if grayscale images exist: 
    if(len(os.listdir(pathUsed)) == 0): 
        print('--------------------------------------------------------')
        print('Creating grayscale images....')
        print('--------------------------------------------------------')
        files = [f for f in os.listdir(imagePath) if os.path.isfile(os.path.join(imagePath,f))]
        imgs = np.empty(len(files), dtype=object)
        grayImgs = np.empty(len(imgs), dtype=object)
        
        for i in range(0, len(imgs)): 
            imgs[i] = cv2.imread(os.path.join(imagePath,files[i]))
            grayImgs[i] = cv2.cvtColor(imgs[i],cv2.COLOR_BGRA2GRAY)
            filename = "{}grayimg{}.jpg".format(pathUsed,i)
            cv2.imwrite(filename, grayImgs[i])

    #read in grayscale files directly if checks pass: 
    files = [f for f in os.listdir(pathUsed) if os.path.isfile(os.path.join(pathUsed,f))]
    imgs = np.empty(len(files), dtype=object)
    # color_files = [f for f in os.listdir(imagePath) if os.path.isfile(os.path.join(imagePath,f))]
    # color_imgs = np.empty(len(color_files), dtype=object)

    for i in range(0, len(imgs)): 
        imgs[i] = cv2.imread(os.path.join(pathUsed,files[i]), 0) #need 0 flag to read as grayscale
        # color_imgs = cv2.imread(os.path.join(imagePath,color_files[i]))
    """
    note:  
    format of imgs:    imgs[frame#][x-cord,y-cord]
    0-indexed 
    """


    threshold = 10

    #make arrays 1-indexed b/c we don't use the 1st frame
    mskImgs=[0]
    mvImgs=[0]
    diffImg = np.empty(len(imgs),dtype=object)

    # operator = np.matrix([-1,0,1])
    for j in range(0,len(imgs)):
        #check that first and last imgs aren't being used
        try: 
            if ((j > 1) and (j <= len(imgs))): 
                # difference between -1 and 1 img; curr image = 0: 
                delta = abs(imgs[j]-imgs[j-2])
                diffImg[j-1] = delta

                #####trying to do the threshold change for the light:  ####
                if((j>15) and (j<len(imgs))):
                    thresh_change = abs(imgs[j][0,0] - imgs[j-15][0,0])
                    print(imgs[j][0,0])
                    print(thresh_change)
                    print('---------------------------------')
                    if imgs[j][0,0] >= 130 and thresh_change > 40 and j > 200:            
                        print(threshold)
                        threshold=threshold+thresh_change
                        print(threshold)
                del thresh_change #memory help
                mask = thresholding(delta,threshold)
                mvImgs.append(np.multiply(delta, mask))
                mskImgs.append(mask)
                
        except IndexError as e: 
            print(e)

    #removes first and last frames that are equal to None: 
    newDiffImgs = []
    for i in range(0,len(diffImg)): 
        if diffImg[i] is not None: 
            newDiffImgs.append(diffImg)


   #testing if mask works: 
    # for i in range(0,len(imgs)-2): 
    #     plt.subplot(2,2,1)
    #     plt.imshow(imgs[i],cmap='gray')
    #     plt.title('original grayscale img')
    #     plt.subplot(2,2,2)
    #     plt.imshow(diffImg[i],cmap='gray')
    #     plt.title('diff img')
    #     plt.subplot(2,2,3)
    #     plt.imshow(mskImgs[i],cmap='gray')
    #     plt.title('mask img')
    #     plt.subplot(2,2,4)
    #     plt.imshow(mvImgs[i],cmap='gray')
    #     plt.title('movement img')
    #     plt.show()

    


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
    
