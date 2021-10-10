#GOAL: detect moving object by looking at large gradients in the temporal evolution of pixael values
import cv2 
import numpy as np 
from scipy import signal

def process_frame(frame): 
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    oneDimOperator = np.array([-1,0, 1])
    filtImage = signal.convolve2d(grayFrame, oneDimOperator)
    return filtImage

def main():    
    #read img frames + convert to grayscale
    video_path = './Videos/Office.mp4'
    # video_path = '../Videos/RedChair.mp4'
    cap = cv2.VideoCapture(video_path)
    if(cap.isOpened()==False):
        print("Error opening video")
    while(cap.isOpened()):
        ret,frame=cap.read()
        if ret == True: 
            frame = process_frame(frame)
            cv2.imshow('Project 1', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
        else: 
            break
    cap.release()
    cv2.destroyAllWindows()

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
    
