from tkinter import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
import argparse 
from numba import jit
import math
import scipy.interpolate as spi
from scipy.interpolate import interp1d
from scipy.interpolate import BSpline, make_interp_spline
#from imageio import imread
import scipy.ndimage as ndimage

#convert video from folder to image
def extractImages(pathIn, pathOut):
    """
    Given a location to retreieve a video and location to store the video
    """
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count))    # change(count*250) and add a way to define a number of frames
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1

def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr

def rotate(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def crop_image(imCrop):
    # Crop image
    rio = cv2.selectROI(img, fromCenter= False)
    imCrop = img[int(rio[1]):int(rio[1]+rio[3]), int(rio[0]):int(rio[0]+rio[2])]

    # Display cropped image
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)

def poly_crop(crop):
    # original image
    # -1 loads as-is so if it will be 3 or 4 channel as the original
    image = crop
    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([[(10,10), (300,300), (10,300)]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def getCurvature(coordinate): #function would     take the contour and how many points along the contour to take
    x_t = np.gradient(coordinate[0])
    y_t = np.gradient(coordinate[1])

    vel = np.array([ [x_t[i], y_t[i]] for i in range(np.size(x_t))])

    print(vel)

    speed = np.sqrt(x_t * x_t + y_t * y_t)

    print(speed)

    tangent = np.array([1/speed] * 2).transpose() * vel

    print(tangent)

    ss_t = np.gradient(speed)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)

    curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5

    print("curvature")
    #curvature_val = np.where(curvature_val != [0])
    return curvature_val                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

def getCurvature2(coordinate): 
    #first derivatives 
    dx= np.gradient(coordinate[:,0])
    dy = np.gradient(coordinate[:,1])

    #second derivatives 
    dxx = np.gradient(dx)
    dyy = np.gradient(dy)

    #calculation of curvature from the typical formula
    curvature_val = np.abs(dx * dyy - dxx * dy) / (dx * dx + dy * dy)**1.5

    print("curvature2")
    return curvature_val

refPt =     []
cropping = False

#define the paths to get the video where to store the frames
input_loc = r'C:\Users\acey2\OneDrive\Documents\GitHub\curvature-detection\Video\20220127_003218.mp4'
output_loc = r'C:\Users\acey2\OneDrive\Documents\GitHub\curvature-detection\Frame'
img = cv2.imread(r'C:\Users\acey2\OneDrive\Documents\GitHub\curvature-detection\Image\2.png')



'''click and cropping the curve'''
"""imCrop = crop_image(img)

crop = cv2.imread('test.png', -1)

cropped_image = poly_crop(crop)
# save the result
cv2.imwrite('cropped_image.png', cropped_image)"""

'''c2 = getCurvature2(data)
c2 = np.where(c2 != [0])
print(c2)
print("hello")

plt.figure(3)
t = np.linspace(0,(np.size(c2)-1), np.size(c2)) 
plt.plot(t,c2[1])
plt.show()'''

'''
#use B spline
#use cublic permine spline psypie
#spiral search'''

#hough transf
