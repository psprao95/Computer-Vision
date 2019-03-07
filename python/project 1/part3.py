import cv2
import numpy as np
import sys

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

cv2.imshow("input image: " + name_input, inputImage)

rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))


def limit(b,g,r):
    input=[b,g,r]
    output=[]
    for i in input:
        if i<0:
            i=0
        elif i>1:
            i=1
        output.append(i)
    return output

def limitRGB(b,g,r):
    input=[b,g,r]
    output=[]
    for i in input:
        if i<0:
            i=0
        elif i>255:
            i=255
        output.append(i)
    return output
# Conversion functions

def sRGB_To_NonLinearRGB(b,g,r):
    return limit(b/255,g/255,r/255)


def NonLinearRGB_To_LinearRGB(b, g, r):
    input = [b, g, r]
    output=[]
    for i in input:
        if i<0.03928:
            output.append(i/12.92)
        else:
            output.append(pow((i+0.055)/1.055,2.4))
    return limit(output[0],output[1],output[2])


def LinearRGB_To_XYZ(b,g,r):
    x = 0.412453*r + 0.35758*g + 0.180423*b
    y = 0.212671*r + 0.71516*g + 0.072169*b
    z = 0.019334*r + 0.119193*g + 0.950227*b

    return [x,y,z]



def XYZ_To_LinearRGB(x,y,z):
     r = 3.240479*x - 1.53715*y - 0.498535*z
     g = -0.969256*x + 1.875991*y + 0.041556*z
     b = 0.055648*x - 0.204043*y + 1.057311*z

     return [b,g,r]


def LinearRGB_To_NonLinearRGB(b,g,r):
    input=[b,g,r]
    output=[]
    for i in input:
        if i<0.00304:
            output.append(12.92*i)
        else:
            output.append((1.055*pow(i,1/2.4))-0.055)

    return limit(output[0],output[1],output[2])

def NonLinearRGB_To_sRGB(b,g,r):
    return limitRGB(b*255,g*255,r*255)


def XYZ_To_xyY(x,y,z):
    a=x/(x+y+z)
    b=y/(x+y+z)
    return [a,b,y]


def xyY_To_XYZ(x,y,Y):
    if(y==0):
        return [0,0,Y]
    X=(x/y)*Y
    Z=(1-x-y)*Y/y
    return [X,Y,Z]

# end of example of going over window

temp = np.copy(inputImage).astype(np.float32)

for i in range (H1,H2+1):
    for j in range (W1,W2+1):
        b,g,r=inputImage[i,j]
        temp[i,j]=sRGB_To_NonLinearRGB(b,g,r)

        b,g,r=temp[i,j]
        temp[i,j]=NonLinearRGB_To_LinearRGB(b,g,r)

        b,g,r=temp[i,j]
        temp[i,j]=LinearRGB_To_XYZ(b,g,r)

        X,Y,Z=temp[i,j]
        temp[i,j]=XYZ_To_xyY(X,Y,Z)

Ymin=1
Ymax=0
for i in range (H1,H2+1):
    for j in range (W1,W2+1):
        x,y,Y=temp[i,j]
        if Y<Ymin:
            Ymin=Y
        if Y>Ymax:
            Ymax=Y

for i in range (H1,H2+1):
    for j in range (W1,W2+1):
        x,y,Y=temp[i,j]
        Yscaled=(Y-Ymin)*1/(Ymax-Ymin)
        temp[i,j]=[x,y,Yscaled]


for i in range (H1,H2+1):
    for j in range (W1,W2+1):
        L,u,v=temp[i,j]
        temp[i,j]=xyY_To_XYZ(L,u,v)

        x,y,z=temp[i,j]
        temp[i,j]=XYZ_To_LinearRGB(x,y,z)

        b,g,r=temp[i,j]
        temp[i,j]=LinearRGB_To_NonLinearRGB(b,g,r)

        b,g,r=temp[i,j]
        temp[i,j]=NonLinearRGB_To_sRGB(b,g,r)


output_image = np.copy(inputImage).astype(np.uint8)

for i in range (H1,H2+1):
    for j in range (W1,W2+1):
        output_image[i,j]=temp[i,j]

cv2.imshow("output",output_image)
cv2.imwrite(name_output,output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
