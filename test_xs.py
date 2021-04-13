import cv2 #khai bao thu vien open cv
from matplotlib import pyplot as plt
import numpy as np
kernel = np.ones((5,5),np.uint8)

def show_img_simple(img,cmap=None):
    plt.imshow(img,cmap=cmap)
    plt.xticks([]), plt.yticks([])
    plt.show()


def show_img(img1,img2,img3,img4,img5,img6):

    titles = ['img_origin', 'img_gray','blur','blur','median','blgfilter']
    images = [img1, img2, img3, img4, img5, img6]
    
    for i in range(6):
        plt.subplot(2, 3, i+1), plt.imshow(images[i],cmap='gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()


if __name__ == '__main__':

    img = cv2.imread('12.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img_median = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #######################
    dst = cv2.filter2D(img, -1, kernel)
    blur = cv2.blur(img, (5, 5))
    gblur = cv2.GaussianBlur(img, (5, 5), 0)
    median = cv2.medianBlur(img, 5)
    #######################
  
    # img_darker = cv2.equalizeHist(img_gray)
    blur = cv2.blur(img_gray, (5, 5))# loc noise
    bilateralFilter = cv2.bilateralFilter(blur, 4, 75, 75)# loc noise

    #######################
    ret,img_threshold1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    ret,img_threshold2 = cv2.threshold(bilateralFilter,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    ##########################
    show_img(img,img_gray,blur,bilateralFilter,img_threshold1,img_threshold2)
    # cv2.imshow('img_origin',img_median)

    cv2.waitKey(0)