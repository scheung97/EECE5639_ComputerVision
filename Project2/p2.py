import cv2 
import numpy as np
import math




def main(): 
    print("Hello World") 
    img = cv2.imread("test.jpg")
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    angle = math.radians(90)
    rotate_z = np.array([[math.cos(angle), -1*math.sin(angle)],[math.sin(angle),math.cos(angle)]])

    print(grey_img.shape)
    for x in range(len(grey_img[0])):
        for y in range(len(grey_img[1])):
                test = grey_img[x][y]*rotate_z


if __name__ == "__main__":
    main()



"""
extra credit info: 
https://learnopencv.com/homography-examples-using-opencv-python-c/
"""