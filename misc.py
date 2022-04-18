
'''
#Finger detection color range
hsvim = cv2.cvtColor(imCrop, cv2.COLOR_BGR2HSV) #changes the color
lower = np.array(50, dtype = "uint8") #lower bound of color of the finger
upper = np.array(255, dtype = "uint8") #lower bound of color of the finger
skinRegionHSV = cv2.inRange(hsvim, lower, upper) #sets the range from the above value
blurred = cv2.blur(skinRegionHSV, (2,2)) #blurs the image(not sure)
ret,edges = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY) #Makes a clear contrast between the finger and everything else together
cv2.imshow("thresh", edges)
#use polycrop in order to crop out certain parts of the image
'''


'''
x_data = []
y_data = []
xy = []
j_first = False
j = 0
i = 0
while(i < len(window)):
    j=0
    while (j < len(window[i]) or j_first == False):
        if window[i][j] == 255:
            x_data.append(i)
            y_data.append(j)
            xy.extend((i, j))
            j_first = True
            print(window[i][j])
        j = j+1    
    j_first = False
    i = i+1
'''
'''
# Reshape into [[x1, y1],...]
data = np.array(xy).reshape((-1, 2))
# Translate points back to original positions.
data[:, 1] = bounds[1] - data[:, 1]
   
plt.figure(1, figsize=(8, 16))
ax1 = plt.subplot(211)
ax1.imshow(edges)
ax2 = plt.subplot(212)
ax2.axis([0, edges.shape[1], edges.shape[0], 0])
ax2.plot(data[:,1])
plt.show()
'''

'''
plt.figure(1, figsize=(8, 16))
ax1 = plt.subplot(211)
ax1.imshow(edges)
ax2 = plt.subplot(212)
ax2.axis([0, edges.shape[1], edges.shape[0], 0])
y_t = np.linspace(0,np.size(y_data)-1,np.size(y_data))
ax2.plot(x_data, y_data)
plt.show()

xdata = data[:,0]
ydata = data[:,1]

z = np.polyfit(xdata, ydata, 5)
f = np.poly1d(z)

t = np.arange(0, edges.shape[1], 1)
plt.figure(2, figsize=(8, 16))
ax1 = plt.subplot(211)
ax1.imshow(edges)
ax2 = plt.subplot(212)
ax2.axis([0, edges.shape[1], edges.shape[0], 0])
ax2.plot(t, f(t))
print("hello")
'''

'''
#polyline cropping method
def shape_selection(event, x, y, flags, param):
    global ref_point, cropping
    if event == cv2.EVENT_LBUTTONDOWN: #start recording 
        ref_point = [(x, y)]
        cropping = True
    #    break
    
    if event == cv2.EVENT_LBUTTONUP: #take another point by pressing the left mouse button
        ref_point.append((x, y))
        cropping = True
    
    elif event == cv2.EVENT_RBUTTONUP: # check to see if the left mouse button was released
        ref_point.append((x, y))
        cropping = False

    ref_points = ref_points.reshape((-1,1,2))
    cv2.polylines(img,[ref_points],True,(0,255,255))
    cv2.imshow("image", img)
'''

'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--img", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(argparse["img"])

clone = img.copy()
cv2.namedWindow("img")
cv2.setMouseCallback("img", shape_selection)

while True:# keep looping until the 'q' key is pressed
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    elif key == ord("c"):# if the 'c' key is pressed, break from the loop
    #break


#splining

#curvature after splining

# #getting the contour the system
# contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img, contours, -1, (0,255,0), 20)
# cv.imshow("contours", img)

# print(len(contours))

#indices = np.where(edges != [0])

'''


'''
#Applying blue and Canny
img3 = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)
img3 = cv2.GaussianBlur(img3,(5,5),0)
edges = cv2.Canny(img3,100,200)
#ret,edges = cv2.threshold(imCrop,0,255,cv2.THRESH_BINARY)
cv2.imshow("Image", edges)
print(edges)

plt.figure(1)
plt.plot(1),plt.imshow(edges)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

ret, thresh = cv2.threshold(edges, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("Image", thresh)
#defining the bounds of the box
#very needs to be tuned for each incoming image
bounds = [rio[1], rio[3]]
# Now the points we want are the lowest-index 255 in each row
window = edges[bounds[1]:bounds[0]:-1].transpose()

'''

'''
#Old search code
@jit(nopython=True)
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1

xy = []
for i in range(len(window)):
    col = window[i]
    j = find_first(255, col)
    if j != -1:
        xy.extend((i, j))
# Reshape into [[x1, y1],...]
data = np.array(xy).reshape((-1, 2))
# Translate points back to original positions.
data[:, 1] = bounds[1] - data[:, 1]
'''

#vertical search

'''
ROOT = "C:\Users\acey2\OneDrive\Documents\GitHub"
image_folder = ROOT + "curvature-detection/Image/"
image = image_folder + "2.png"
'''