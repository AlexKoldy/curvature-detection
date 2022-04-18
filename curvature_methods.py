import numpy as np

class Curvature:
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