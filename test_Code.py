import cv2
import numpy as np
from matplotlib import pyplot as plt



def show_img_simple(img,cmap=None):
    plt.imshow(img,cmap=cmap)
    plt.xticks([]), plt.yticks([])
    plt.show()

# def nothing(x):
#     pass

# cv2.namedWindow("Tracking")
# cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
# cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
# cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

def show_img_simple(img,cmap=None):
    plt.imshow(img,cmap=cmap)
    plt.xticks([]), plt.yticks([])
    plt.show()


def show_img(img1):

    titles = ['img_origin']
    images = [img1]
    
    for i in range(1):
        plt.subplot(1, 1, i+1), plt.imshow(images[i],cmap='gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.show()

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
while True:
    
    frame = cv2.imread('12.png')
    
    blur = cv2.blur(frame, (5, 5))# loc noise
    bilateralFilter = cv2.bilateralFilter(blur, 2, 75, 75)
    
    hsv = cv2.cvtColor(bilateralFilter, cv2.COLOR_BGR2HSV)

    # l_h = cv2.getTrackbarPos("LH", "Tracking")
    # l_s = cv2.getTrackbarPos("LS", "Tracking")
    # l_v = cv2.getTrackbarPos("LV", "Tracking")

    # u_h = cv2.getTrackbarPos("UH", "Tracking")
    # u_s = cv2.getTrackbarPos("US", "Tracking")
    # u_v = cv2.getTrackbarPos("UV", "Tracking")




    l_b = np.array([12, 0, 0])
    u_b = np.array([255, 255, 255])


    mask = cv2.inRange(hsv, l_b, u_b)


    res = cv2.bitwise_and(frame, frame, mask=mask)
    _, gray_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    erosion1 = cv2.erode(gray_mask,kernel,iterations = 1)
    
    opening = cv2.morphologyEx(erosion1, cv2.MORPH_OPEN, kernel,iterations=3)

    

    # opening2 = cv2.morphologyEx(erosion1, cv2.MORPH_CLOSE, kernel,iterations=3)
    
#####################################################################
    sure_img = cv2.dilate(opening,kernel,iterations=1)
######################################################################
    # contours, hierarchy = cv2.findContours(sure_img, cv2.RETR_TREE+cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#######################################################################
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)

    sure_fg = np.uint8(sure_fg) 
    unknown = cv2.subtract(sure_img,sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
# Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(bilateralFilter,markers)
    
    # show_img_simple(markers,cmap='gray')
    # print(markers.shape, markers.dtype)
    contours, hierarchy = cv2.findContours(markers.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    total_contours=0
    _x = 0
    _y = 0

    for _i in contours:

        (x,y,w,h) = cv2.boundingRect(_i)
        area = cv2.contourArea(_i)
        if (abs (x - _x) < 4) or area >= 10000 or abs (y - _y) == 2 :
            continue

        total_contours+=1
        cv2.putText(frame,"{}".format(str(total_contours)), (x+w//2,y+h//2),cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,0),1)
        cv2.drawContours(frame, [_i], -1, (255, 0, 0), 2)
        
        cv2.waitKey(0)
        # total_contours+=1
        print("x:{} and y:{}".format(x,y))
        print("_x:{} and _y:{}".format(_x,_y))
        print("area:{}".format(area))
        _x = x
        _y = y

        cv2.imshow("frame", frame)
    print("number of contour =" + str(total_contours))
            
   
    # cv2.imshow("opening", opening)

    # show_img(opening)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()

