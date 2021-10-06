#GOAL: detect moving object by looking at large gradients in the temporal evolution of pixael values
import cv2 
import numpy as np 
import os 

def make_video(): 
    print('video created')
def main():    
    #read img frames + convert to grayscale
    officePath = '../Office'
    redChairPath = r'../RedChair'

    onlyfiles = [f for f in os.listdir(officePath) if os.path.isfile(os.path.join(officePath,f))]
    images = np.empty(len(onlyfiles), dtype=object)
    gray_images = np.empty(len(images), dtype=object)

    for i in range(0, len(onlyfiles)): 
        images[i] = cv2.imread(os.path.join(officePath,onlyfiles[i]))
        gray_images[i] = cv2.cvtColor(images[i],cv2.COLOR_BGRA2GRAY)

        """ test for if image conversion worked:"""
        #filename = "../OfficeGray/grayimg%i.jpg"%i 
        #cv2.imwrite(filename, gray_images[i])
        # if(i <= 5): 
        #     cv2.imshow('gray', gray_images[i])
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()


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
    
