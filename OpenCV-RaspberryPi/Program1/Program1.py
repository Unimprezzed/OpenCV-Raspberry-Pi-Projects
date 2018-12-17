import cv2
import numpy as np
import math
import sys
#Clips input to be within a minimum and maximum range. 
def clip(x, _min, _max):
    return max(min(x, _max),_min)
#Inverse gamma function
def inv_gamma(v):
    return v / 12.92 if v < 0.03928 else math.pow( ((v + 0.055)/1.055), 2.4)
#Gamma function
def gamma(d):
    return 12.92*d if d < 0.00304 else 1.055*math.pow(d, (1/2.4)) - 0.055
#Linear scale
def linear_scale(x, a, b, A, B):
    return A +((x-a)*(B-A))/(b-a)
#Converts BGR (the way OpenCV stores color information) to RGB 
def BGR_to_RGB(BGR):
    R,G,B = BGR[2],BGR[1],BGR[0]
    return [R,G,B]

#Converts RGB to BGR
def RGB_to_BGR(RGB):
    B,G,R = RGB[2],RGB[1],RGB[0]
    return [B,G,R]

#Transforms an RGB pixel to Luv
def sRGB_to_Luv(RGB):
    #Convert sRGB to XY
    _R,_G,_B = float(RGB[0])/255.0, float(RGB[1])/255.0, float(RGB[2])/255.0
    R,G,B = inv_gamma(_R),inv_gamma(_G),inv_gamma(_B)
    _RGB = np.mat([R,G,B]).transpose()
    M = np.mat([[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]])
    XYZ = np.reshape(np.array(M*_RGB), (3))
    #XYZ to Luv
    X,Y,Z = XYZ[0],XYZ[1],XYZ[2]
    X_w, Y_w, Z_w = 0.95, 1.0, 1.09
    u_w = 4*X_w / (X_w + 15*Y_w + 3*Z_w)
    v_w = 9*Y_w / (X_w + 15*Y_w + 3*Z_w)
    t = Y / Y_w
    L = (116*(t**(1/3))) - 16 if t > 0.008856 else 903.3*t
    d = X + 15*Y + 3*Z
    _u = 4*X / d if d != 0 else 0.0
    _v = 9*Y/ d if d != 0 else 0.0
    u = 13 * L * (_u - u_w)
    v = 13*L*(_v - v_w)
    Luv = np.array([L,u,v])
    return Luv

def Luv_to_sRGB(Luv):
    #convert from Luv to XYZ
    L,u,v = Luv[0],Luv[1],Luv[2]
    X_w, Y_w, Z_w = 0.95, 1.0, 1.09
    u_w = 4*X_w / (X_w + 15*Y_w + 3*Z_w)
    v_w = 9*Y_w / (X_w + 15*Y_w + 3*Z_w)
    _u = (u + 13*u_w *L)/(13*L) if L != 0 else 0.0
    _v = (v + 13*v_w*L)/(13*L) if L != 0 else 0.0
    Y = (((L + 16)/116)**3)*Y_w if L > 7.9996 else (L / 903.3)*Y_w
    X,Z = 0,0
    if _v != 0.0:
        X = Y*2.25*(_u/_v)
        Z = (Y*(3 - 0.75*_u - 5*_v))/_v
    M = np.mat([[3.240479, -1.53715, -0.498535],[-0.969256,1.875991, 0.041556],[0.055648, -0.204043, 1.057311]])
    XYZ = np.mat([X,Y,Z]).transpose()
    RGB = np.reshape(np.array(M*XYZ), (3))
    RGB = [clip(x, 0.0, 1.0) for x in RGB]
    RGB = [gamma(x) for x in RGB]
    RGB = [np.uint8(round(x * 255.0)) for x in RGB]
    return RGB

#Transforms an RGB pixel to xyY
def sRGB_to_xyY(RGB):
    #Convert sRGB to XY
    _R,_G,_B = float(RGB[0])/255.0, float(RGB[1])/255.0, float(RGB[2])/255.0
    R,G,B = inv_gamma(_R),inv_gamma(_G),inv_gamma(_B)
    _RGB = np.mat([R,G,B]).transpose()
    M = np.mat([[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]])
    XYZ = np.reshape(np.array(M*_RGB), (3))
    #XYZ to xyY
    X,Y,Z = XYZ[0],XYZ[1],XYZ[2]
    x = X / (X + Y + Z) if (X + Y + Z) != 0.0 else 0.0
    y = Y / (X + Y + Z) if (X + Y + Z) != 0.0 else 0.0
    xyY = [x,y,Y]
    return xyY

def xyY_to_sRGB(xyY):
    #convert from xyY to XYZ
    x,y,Y = xyY[0],xyY[1],xyY[2]
    X = (x/y)*Y if y != 0.0 else 0.0
    Z = ((1.0 - x - y)/y)*Y if y != 0.0 else 0.0
    M = np.mat([[3.240479, -1.53715, -0.498535],[-0.969256,1.875991, 0.041556],[0.055648, -0.204043, 1.057311]])
    XYZ = np.mat([X,Y,Z]).transpose()
    RGB = np.reshape(np.array(M*XYZ), (3))
    RGB = [clip(_element, 0.0, 1.0) for _element in RGB]
    RGB = [gamma(_element) for _element in RGB]
    RGB = [np.uint8(round(_element * 255.0)) for _element in RGB]
    return RGB

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print(sys.argv[0] + ": got " + str(len(sys.argv) - 1) + " arguments. Expected two: <width> <height>")
    exit(-1)
  width = int(sys.argv[1])
  height = int(sys.argv[2])
  img1 = np.zeros((height, width,3), dtype=np.uint8).tolist()
  img2 = np.zeros((height, width,3),dtype=np.uint8).tolist()
  
  print("Creating images in Luv and xyY")
  for i in range(0, height):
      for j in range(0,width):
          x = float(j)/float(width)
          y = float(i)/float(height)
          Y = 1.0
          L = 90.0
          u = x*512.0 - 255.0
          v = y*512.0 - 255.0
          Luv = [L,u,v]
          xyY = [x,y,Y]
          srgb_2 = xyY_to_sRGB(xyY)
          srgb_1 = Luv_to_sRGB(Luv)
          img1[i][j] = RGB_to_BGR(srgb_1)
          img2[i][j] = RGB_to_BGR(srgb_2)
  img1 = np.array(img1)
  img2 = np.array(img2)
  print("Converting to sRGB...")
  cv2.imwrite('xyY.png', img2)
  cv2.imwrite('Luv.png', img1)
  cv2.imshow('xyY',img2)
  cv2.imshow('Luv',img1)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
