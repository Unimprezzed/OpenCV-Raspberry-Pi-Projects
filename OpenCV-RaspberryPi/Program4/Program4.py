import cv2
import numpy as np
import math
import sys

def clip(x, _min, _max):
    return max(min(x, _max),_min)
def inv_gamma(v):
    return v / 12.92 if v < 0.03928 else math.pow( ((v + 0.055)/1.055), 2.4)
#Gamma function
def gamma(d):
    return 12.92*d if d < 0.00304 else 1.055*math.pow(d, (1/2.4)) - 0.055
#Inverse gamma function
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

#Runs the operation on a window
def runOnWindow(w1, h1, w2, h2, inImage, outName):
    rows,cols = inImage.shape[1], inImage.shape[0]
    label = "Window: [(" + str(w1) + ", " + str(h1) + "),(" + str(w2) + ", " + str(h2) + ")]"
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    print("Image recieved is of size " + str(inImage.shape))
    in_image = inImage.astype(float).tolist()
   
    im_xyY = np.zeros((cols,rows,3)).tolist()
    print("Converting to xyY...")
    for i in range(0, inImage.shape[0]):
        for j in range(0, inImage.shape[1]):
            in_px = [in_image[i][j][2], in_image[i][j][1],in_image[i][j][0]]
            xyY_px = sRGB_to_xyY(in_px)
            im_xyY[i][j] = xyY_px
    print("Performing linear scaling...")
    im_xyY = np.array(im_xyY)
    window = im_xyY[h1:h2,w1:w2]
    f_W = window[:,:,2].flatten()
    wY_low = np.amin(f_W)
    wY_high = np.amax(f_W)
    print("Min Y in window: " + str(wY_low))
    print("Max Y in window: " + str(wY_high))
    out_image = np.zeros((cols, rows, 3)).tolist()
    print("Applying linear scaling to image in xyY color space...")
    for i in range(0, cols):
        for j in range(0, rows):
            in_px = in_image[i][j]
            Y = im_xyY[i][j][2]
            Y = clip(Y, wY_low, wY_high)
            Y = linear_scale(Y, wY_low, wY_high, 0.0, 1.0)
            im_xyY[i][j][2] = Y
            out_px = xyY_to_sRGB(im_xyY[i][j])
            out_image[i][j] = RGB_to_BGR(out_px)
            
    print("Done! Producing output image...")
    out_image= np.array(out_image,dtype=np.uint8)
    cv2.putText(out_image,label,(2,rows-40),font,0.5,(255,0,255),1,cv2.LINE_8)
    cv2.imshow('output',out_image)
    cv2.imwrite(outName,out_image)
    
if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(sys.argv[0] + ": got " + str(len(sys.argv) - 1) + " arguments. Expected six: <w1> <h1> <w2> <h2> <ImageIn> <ImageOut>")
        exit(-1)
    w1 = float(sys.argv[1])
    h1 = float(sys.argv[2])
    w2 = float(sys.argv[3])
    h2 = float(sys.argv[4])
    inputname = sys.argv[5]
    outName = sys.argv[6]
    if w1 < 0 or h1 < 0 or w2 <= w1 or h2 <= h1 or w2 > 1 or h2 > 1:
        print("arguments must satisfy 0 <= w1 < w2 <= 1 \n, 0 <= h1 < h2 < 1")
        exit(-1)
    inImage = cv2.imread(inputname,cv2.IMREAD_UNCHANGED)
    if inImage is None:
        print("Could not open or find the image " + inputname)
        exit(-1)
    #Equivalent to check in original program.
    if inImage.dtype != np.uint8:
        print( inputname + " is not a standard color image ")
        exit(-1)
    rows = inImage.shape[1]
    cols = inImage.shape[0]
    _w1 = int(w1*(cols)) #Starting position x 
    _h1 = int(h1*(rows)) #Starting position y
    _w2 = int(w2*(cols)) #Ending position x
    _h2 = int(h2*(rows)) #Ending position y   
    #Dynamically allocate the arrays of size rows x cols
    runOnWindow(_w1, _h1, _w2, _h2, inImage, outName)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Goodbye...")
