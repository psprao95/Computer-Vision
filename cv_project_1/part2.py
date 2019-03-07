import cv2
import numpy as np
import sys
import math

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


# function to limit linear and non linear RGB values to (0,1) range
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

# function to limit sRGB values to (0,255) range
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


# conversion from sRGB to Non-linear RGB
def sRGB_To_NonLinearRGB(b,g,r):
    return limit(b/255,g/255,r/255)


# conversion from nonlinear RGB to linear RGB
def NonLinearRGB_To_LinearRGB(b, g, r):
    input = [b, g, r]
    output=[]
    for i in input:
        if i<0.03928:
            output.append(i/12.92)
        else:
            output.append(pow((i+0.055)/1.055,2.4))
    return limit(output[0],output[1],output[2])


# conversion from linear RGB to XYZ
def LinearRGB_To_XYZ(b,g,r):
    x = 0.412453*r + 0.35758*g + 0.180423*b
    y = 0.212671*r + 0.71516*g + 0.072169*b
    z = 0.019334*r + 0.119193*g + 0.950227*b

    return [x,y,z]


# conversion from  XYZ to Luv
def XYZ_To_Luv(x,y,z):
    xw=0.95
    yw=1
    zw=1.09
    uw=(4*xw)/(xw+ 15*yw+ 3*zw)
    vw=(9*yw)/(xw+ 15*yw +3*zw)

    t=y/yw

    if t>0.008856:
        L = 116*pow(t,1/3) - 16
    else:
        L=903.3*t

    d=x+ 15*y +3*z
    if d==0:
        return [L,0,0]
    up=4*x/d
    vp=9*y/d
    u=13*L*(up-uw)
    v=13*L*(vp-vw)
    if L<0:
        L=0
    elif L>255:
        L=255

    return [L,u,v]


# conversion from Luv to XYZ
def Luv_To_XYZ(L,u,v):

    if L==0:
        return [0,0,0]
    xw=0.95
    yw=1.0
    zw=1.09
    uw=(4*xw)/(xw + 15*yw + 3*zw)
    vw=(9*yw)/(xw + 15*yw + 3*zw)

    up=(u + 13*uw*L)/(13*L)
    vp=(u + 13*vw*L)/(13*L)


    if L>7.9996:
        y=pow((L+16)/116,3)*yw
    else:
        y=(L*yw)/903.3

    if vp==0:
        return [0,y,0]
    else:
        x=y*2.25*(up/vp)
        z=y*(3 - 0.75*up - 5*vp)/vp
        return [x,y,z]


# conversion from   XYZ to linear RGB
def XYZ_To_LinearRGB(x,y,z):
     r = 3.240479*x - 1.53715*y - 0.498535*z
     g = -0.969256*x + 1.875991*y + 0.041556*z
     b = 0.055648*x - 0.204043*y + 1.057311*z

     return [b,g,r]


# conversion from linear RGB to non-linear RGB
def LinearRGB_To_NonLinearRGB(b,g,r):
    input=[b,g,r]
    output=[]
    for i in input:
        if i<0.00304:
            output.append(12.92*i)
        else:
            output.append((1.055*pow(i,1/2.4))-0.055)

    return limit(output[0],output[1],output[2])


# conversion from non-linear RGB to sRGB
def NonLinearRGB_To_sRGB(b,g,r):
    return limitRGB(b*255,g*255,r*255)



# copy the input image
temp = np.copy(inputImage).astype(np.float32)


# conversion of pixels in window to Luv domain
for i in range (H1,H2+1):
    for j in range (W1,W2+1):
        b,g,r=inputImage[i,j]
        temp[i,j]=sRGB_To_NonLinearRGB(b,g,r)

        b,g,r=temp[i,j]
        temp[i,j]=NonLinearRGB_To_LinearRGB(b,g,r)

        b,g,r=temp[i,j]
        temp[i,j]=LinearRGB_To_XYZ(b,g,r)

        b,g,r=temp[i,j]
        temp[i,j]=XYZ_To_Luv(b,g,r)




# Performing Histogram Equalization on the L values
LHistogram={}
Frequency={}
for i in range(0,101):
    LHistogram[i]=0

for i in range (H1,H2+1):
    for j in range (W1,W2+1):
        L,u,v=temp[i,j]
        Lrounded=math.floor(L)
        temp[i,j]=[Lrounded,u,v]
        LHistogram[Lrounded]+=1

for i in range(0,101):
    if i==0:
        Frequency[i]=LHistogram[i]
    else:
        Frequency[i]=Frequency[i-1]+LHistogram[i]

for i in range(0,101):
    if(i==0):
        val=(Frequency[i]/2)*(101/Frequency[100])
        LHistogram[i]=math.floor(val)
    else:
        val=((Frequency[i]+Frequency[i-1])/2)*101/Frequency[100]
        LHistogram[i]=math.floor(val)
    if LHistogram[i]>100:
        LHistogram[i]=100
    if(LHistogram[i]<0):
        LHistogram[i]=0

for i in range(H1, H2+1) :
    for j in range(W1, W2+1) :
        L,u,v=temp[i,j]
        temp[i,j]=[LHistogram[L],u,v]


# conversion back to sRGB
for i in range (H1,H2+1):
    for j in range (W1,W2+1):
        L,u,v=temp[i,j]
        temp[i,j]=Luv_To_XYZ(L,u,v)

        x,y,z=temp[i,j]
        temp[i,j]=XYZ_To_LinearRGB(x,y,z)

        b,g,r=temp[i,j]
        temp[i,j]=LinearRGB_To_NonLinearRGB(b,g,r)

        b,g,r=temp[i,j]
        temp[i,j]=NonLinearRGB_To_sRGB(b,g,r)


# writing output image
output_image = np.copy(inputImage).astype(np.uint8)

for i in range (H1,H2+1):
    for j in range (W1,W2+1):
        output_image[i,j]=temp[i,j]

cv2.imshow("output",output_image)
cv2.imwrite(name_output,output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
