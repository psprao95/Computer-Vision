import cv2
import numpy as np

gray_image1=cv2.imread('lenna.bmp');#mistake?
gray_image2=cv2.imread('lenna.bmp',cv2.IMREAD_GRAYSCALE);
gray_image3=cv2.imread('lenna.bmp',cv2.IMREAD_UNCHANGED);
gray_image4=cv2.imread('fruits.jpg',cv2.IMREAD_GRAYSCALE);

cv2.namedWindow('gray1',cv2.WINDOW_AUTOSIZE);
cv2.imshow('gray1',gray_image1);
print("gray_image1",gray_image1[0,0]);

cv2.namedWindow('gray2',cv2.WINDOW_AUTOSIZE);
cv2.imshow('gray2',gray_image2);
print("gray_image2",gray_image2[0,0]);

cv2.namedWindow('gray3',cv2.WINDOW_AUTOSIZE);
cv2.imshow('gray3',gray_image3);
print("gray_image3",gray_image3[0,0]);

cv2.namedWindow('gray4',cv2.WINDOW_AUTOSIZE);
cv2.imshow('gray4',gray_image4);
print("gray_image4",gray_image4[0,0]);

BGR_image1=cv2.imread('fruits.jpg');
BGR_image2=cv2.imread('fruits.jpg',cv2.IMREAD_UNCHANGED)
BGR_image3=cv2.imread('fruits.jpg', cv2.IMREAD_COLOR);

cv2.namedWindow('color1',cv2.WINDOW_AUTOSIZE);
cv2.imshow('color1',BGR_image1);
print("color1",BGR_image1[0,0],"red=",BGR_image1[0,0,2]);

cv2.namedWindow('color2',cv2.WINDOW_AUTOSIZE);
cv2.imshow('color2',BGR_image2);
print("color2",BGR_image2[0,0],"red=",BGR_image2[0,0,2]);

cv2.namedWindow('color3',cv2.WINDOW_AUTOSIZE);
cv2.imshow('color3',BGR_image3);
print("color3",BGR_image3[0,0],"red=",BGR_image3[0,0,2]);

cv2.namedWindow('colorimage',cv2.WINDOW_AUTOSIZE)
RGB_image1=cv2.cvtColor(BGR_image1,cv2.COLOR_BGR2RGB);
cv2.imshow('color4',RGB_image1)
print("color4",RGB_image1[0,0],"blue=",RGB_image1[0,0,2])
cv2.waitKey(0);
cv2.destroyAllWindows;
