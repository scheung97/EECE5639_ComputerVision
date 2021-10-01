#GOAL: detect moving object by looking at large gradients in the temporal evolution of pixael values

def main(){
    #read img frames + convert to grayscale
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
}

       



if __name__ == "__main__": 
    main()
    
