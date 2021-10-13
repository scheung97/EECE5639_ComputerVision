#GOAL: detect moving object by looking at large gradients in the temporal evolution of pixael values
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os

def thresholding(frame, threshold): 
    img = frame.astype(np.uint8)
    th,dst=cv2.threshold(img,threshold, 255,cv2.THRESH_BINARY)
    return dst


def main():    
    ##file paths to images
    officePath = r'../Office/'
    officeGrayPath = r'../OfficeGray/'

    redChairPath = r'../RedChair'
    redChairGrayPath = r'../RedChairGray/'


    ## variables used (update when changing image folders): 
    imagePath = redChairPath 
    pathUsed = redChairGrayPath 
    video = 'redchair' #used for naming output file

    # imagePath = officePath
    # pathUsed = officeGrayPath 
    #  video = 'office' #used for naming output file


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
        
        for i in range(0, len(files)): 
            imgs[i] = cv2.imread(os.path.join(imagePath,files[i]))
            grayImgs[i] = cv2.cvtColor(imgs[i],cv2.COLOR_BGRA2GRAY)
            filename = "{}grayimg_{:004d}.jpg".format(pathUsed,i)
            cv2.imwrite(filename, grayImgs[i])

    #read in grayscale files directly if checks pass: 
    files = [f for f in os.listdir(pathUsed)if os.path.isfile(os.path.join(pathUsed,f))]
    imgs = np.empty(len(files), dtype=object)

    color_files = [f for f in os.listdir(imagePath) if os.path.isfile(os.path.join(imagePath,f))]
    for i in range(0, len(files)): 
        imgs[i] = cv2.imread(os.path.join(pathUsed,files[i]), 0) #need 0 flag to read as grayscale

    """
    note:  
    format of imgs:    imgs[frame#][x-cord,y-cord]
    0-indexed 
    """

    threshold = 10
    #make arrays 1-indexed b/c we don't use the 1st frame
    masks=[]
    count=0
    diffImg = np.empty(len(imgs),dtype=object)
    for j in range(1,len(imgs)):
        #check that first and last imgs aren't being used
        imgs[j]= cv2.GaussianBlur(imgs[j], (3,3), 0)
        try: 
            # difference between -1 and 1 img; curr image = 0: 
            delta = cv2.absdiff(imgs[j],imgs[j-1])
            delta = cv2.dilate(delta, None, iterations=0)
            diffImg[j-1] = delta
            test = imgs[j][235,0] - imgs[0][235,0]
            if j > 500 and test > 40 and count == 0: 
                threshold = threshold + 5
                count = count + 1
            mask = thresholding(delta, threshold)
            masks.append(mask)
            del delta
            del test
        except IndexError as e: 
            print(e)

    mvImgs=[]
    for i in range(1,len(imgs)): 
        mvImg = np.multiply(imgs[i], masks[i-1])
        mvImg=cv2.cvtColor(mvImg, cv2.COLOR_GRAY2BGR)
        mvImgs.append(mvImg)

   #testing if mask works:
    # for i in range(0,len(mvImgs)): 
    #     cv2.imshow('test', mvImgs[i])
    #     cv2.waitKey(0)
    
    output_path =  './Output/{}_output/'.format(video)
    if(os.path.exists(output_path) == False): 
        if(os.path.exists('./Output/')== False): 
            os.mkdir('./Output/')
        os.mkdir('./Output/{}_output/'.format(video))

    for i in range(len(mvImgs)): 
        filename="{}{}_motiondetection_{:004d}.jpg".format(output_path,video,i)
        cv2.imwrite(filename, mvImgs[i])

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
    
