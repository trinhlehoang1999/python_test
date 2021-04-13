import cv2
import numpy as np

kernel = np.ones((3,3),np.uint8)
if (__name__) == '__main__':
    frame = cv2.imread('12.png')
    blur = cv2.blur(frame, (5, 5))# loc noise
    bilateralFilter = cv2.bilateralFilter(blur, 2, 75, 75)
  
    _, gray_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    opening = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)

    erosion1 = cv2.erode(opening,kernel,iterations = 1)

    opening2 = cv2.morphologyEx(erosion1, cv2.MORPH_CLOSE, kernel,iterations=2)
    
#####################################################################
    sure_img = cv2.dilate(opening2,kernel,iterations=3)
######################################################################
    # contours, hierarchy = cv2.findContours(sure_img, cv2.RETR_TREE+cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#######################################################################
    dist_transform = cv2.distanceTransform(opening2,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

    sure_fg = np.uint8(sure_fg) 
    unknown = cv2.subtract(sure_img,sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
# Now, mark the region of unknown with zero
    markers[unknown==250] = 0

    markers = cv2.watershed(bilateralFilter,markers)
    
    # print(markers.shape, markers.dtype)
    contours, hierarchy = cv2.findContours(sure_img, cv2.RETR_CCOMP|cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#########################################################################    
    # print("Number of contours = " + str(len(contours)))
for i in range(len(contours)):
    
    # last column in the array is -1 if an external contour (no contours inside of it)
    if hierarchy[0][i][3] == -1:
        
        # We can now draw the external contours from the list of contours
        cv2.drawContours(frame, contours, i, (255, 0, 0), 4)
        
        
        cv2.imshow("frame", frame)

    # cv2.imshow("frame", frame)

    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()









