import cv2
import numpy as np
import sys
import math

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], "0.2 0.1 0.8 0.5 fruits.jpg out.png")
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
W1 = int(round(w1*(cols-1)))
H1 = int(round(h1*(rows-1)))
W2 = int(round(w2*(cols-1)))
H2 = int(round(h2*(rows-1)))

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

def BGR_NonLinearBGR(b, g, r):
    return clipping(b/255.0, g/255.0, r/255.0)


def NonLinearBGR_LinearBGR(b, g, r):
    color_list = [b, g, r]
    output = []

    for x in color_list :
        if x < 0.03928 :
            output.append(x/12.92)
        else :
            y = pow(((x+0.055)/1.055), 2.4)
            output.append(y)

    return clipping(output[0], output[1], output[2])


def LinearBGR_XYZ(b, g, r):

    x = 0.412453*r + 0.35758*g + 0.180423*b
    y = 0.212671*r + 0.71516*g + 0.072169*b
    z = 0.019334*r + 0.119193*g + 0.950227*b

    return [x, y, z]


def XYZ_LUV(x, y, z):
    Xw = 0.95
    Yw = 1.0
    Zw = 1.09
    Uw = 4*Xw/(Xw + 15*Yw + 3*Zw)
    Vw = 9*Yw/(Xw + 15*Yw + 3*Zw)

    t = y/1.0

    if t > 0.008856 :
        L = 116 * pow(t, (1/3.0)) - 16
    else :
        L = 903.3 * t

    d = x + (15*y) + (3*z)

    if d == 0:
        return [L, 0, 0]
    Up = (4*x)/d
    Vp = (9*y)/d

    u = 13*L*(Up - Uw)
    v = 13*L*(Vp - Vw)

    if L < 0:
        L = 0
    if L > 100:
        L = 100

    return [L, u, v]


def LUV_XYZ(L, u, v):
    Xw = 0.95
    Yw = 1.0
    Zw = 1.09
    Uw = 4*Xw/(Xw + 15*Yw + 3*Zw)
    Vw = 9*Yw/(Xw + 15*Yw + 3*Zw)

    if L == 0:
        return [0, 0, 0]

    u_p = (u + (13*Uw*L))/(13*L)
    v_p = (v + (13*Vw*L))/(13*L)

    if L > 7.9996 :
        y = pow(((L+16)/116.0), 3)
    else :
        y = L/903.3

    if v_p == 0:
        x = 0
        z = 0
    else :
        x = y*2.25*u_p/v_p
        z = y*(3 - 0.75*u_p - 5*v_p)/v_p

    return [x, y, z]


def XYZ_LinearBGR(x, y, z):

    r = 3.240479*x + (-1.53715)*y + (-0.498535)*z
    g = (-0.969256)*x + 1.875991*y + 0.041556*z
    b = 0.055648*x + (-0.204043)*y + 1.057311*z

    return [b, g, r]


def LinearBGR_NonLinearBGR(b, g, r):
    color_list = [b, g, r]
    output = []

    for x in color_list :
        if x < 0.00304 :
            output.append(12.92*x)
        else :
            y = 1.055 * pow(x, (1/2.4)) - 0.055
            output.append(y)


    return clipping(output[0], output[1], output[2])


def NonLinearBGR_BGR(b, g, r):
    return clippingBGR(b*255, g*255, r*255)


def clipping(b, g, r):
    color_list = [b, g, r]
    output = []

    for x in color_list :
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        output.append(x)

    return output


def clippingBGR(b, g, r):
    color_list = [b, g, r]
    output = []

    for x in color_list :
        if x < 0:
            x = 0
        if x > 255:
            x = 255
        output.append(x)

    return output


tmp = np.zeros((rows, cols, 3), dtype=np.float32)

for i in range(H1, H2+1) :
    for j in range(W1, W2+1) :
        tmp[i, j] = inputImage[i, j]


for i in range(H1, H2+1) :
    for j in range(W1, W2+1) :
        b, g, r = inputImage[i, j]

        tmp[i, j] = BGR_NonLinearBGR(b, g, r)

        b, g, r = tmp[i, j]
        tmp[i, j] = NonLinearBGR_LinearBGR(b, g, r)

        b, g, r = tmp[i, j]
        tmp[i, j] = LinearBGR_XYZ(b, g, r)

        x, y, z = tmp[i, j]
        tmp[i, j] = XYZ_LUV(x, y, z)


# Linear Stretching of L

minL = 100.0
maxL = 0.0

for i in range(H1, H2+1) :
    for j in range(W1, W2+1) :
        L, u, v = tmp[i, j]

        if maxL < L:
            maxL = L
        if minL > L:
            minL = L

for i in range(H1, H2+1) :
    for j in range(W1, W2+1) :
        L, u, v = tmp[i, j]
        newL = (L - minL)*100.0/(maxL - minL)
        tmp[i, j] = [newL, u, v]


for i in range(H1, H2+1) :
    for j in range(W1, W2+1) :
        L, u, v = tmp[i, j]
        tmp[i, j] = LUV_XYZ(L, u, v)

        x, y, z = tmp[i, j]
        tmp[i, j] = XYZ_LinearBGR(x, y, z)

        b, g, r = tmp[i, j]
        tmp[i, j] = LinearBGR_NonLinearBGR(b, g, r)

        b, g, r = tmp[i, j]
        tmp[i, j] = NonLinearBGR_BGR(b, g, r)


# end of example of going over window
output_image = np.copy(inputImage).astype(np.uint8)

for i in range(H1, H2+1) :
    for j in range(W1, W2+1) :
        output_image[i, j] = tmp[i, j]

cv2.imshow("output:", output_image)
cv2.imwrite(name_output, output_image);


# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
