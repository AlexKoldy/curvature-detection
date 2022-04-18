import cv2
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import scipy.interpolate as spi
from scipy.interpolate import interp1d
from scipy.interpolate import BSpline, make_interp_spline
#from imageio import imread
import scipy.ndimage as ndimage

#image = "C:\\Users\\acey2\\OneDrive\Documents\\GitHub\\curvature-detection\\Image\\2.png"
#raw = cv2.imread(image)



#raw = cv2.imread("C:/Users/acey2/OneDrive/Documents/GitHub/curvature-detection/Image/2.png")
raw = cv2.imread(r'C:\Users\acey2\OneDrive\Documents\GitHub\curvature-detection\Image\1.png')
#grey = raw[:,:,0]
#threshold = grey>110
#blur = cv2.blur(raw,(2, 2))
blur_img = cv2.GaussianBlur(raw,(5,5),cv2.BORDER_DEFAULT)
edges = cv2.Canny(blur_img,100,200)
cv2.imshow("raw", raw)

kernel_remove_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(2, 1))#extra vertical bits
edged_hor = cv2.erode(edges, kernel_remove_vertical)
cv2.imshow("edges", edged_hor)

# Find coordinates for white pixels
pixels = np.argwhere(edged_hor == 255)
x = (pixels[:, 1])
y = (pixels[:, 0])

# Interpolate with scipy
#f = interp1d(x, y, kind='cubic')
#f = spi.interp1d(x, y, fill_value="extrapolate")
x_new = np.linspace(0, 175, 50)
#y_interp = f(x_new)


plt.figure()
plt.imshow(cv2.cvtColor(edged_hor, cv2.COLOR_BGR2RGB))
#plt.plot(x_new, y_interp)
plt.plot(x, y)
plt.show()
cv2.waitKey()
