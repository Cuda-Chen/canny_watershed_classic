# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

def canny_watershed(filename, sigma, min_edge, ratio):
    # first, read the image
    #image = cv.imread('coins.jpg')
    #image = cv.imread('四破魚(藍圓鰺)2.jpg')
    image = cv.imread(filename)
    cv.imshow('Original image', image)

    '''
    part of misc
    '''
    # change image from BGR to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # apply thresholding to convert the image to binary
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # erode the image
    foreground = cv.erode(thresh, None, iterations = 1)
    # Dilate the image
    backgroundTemp = cv.dilate(thresh, None, iterations = 1)
    # Apply thresholding
    ret, background = cv.threshold(backgroundTemp, 1, 128, 1)
    # Add foreground and background
    marker = cv.add(foreground, background)

    '''
    part of canny
    '''
    # apply (Gaussian) filter for canny edge detector preprocessing
    gaussian = cv.GaussianBlur(marker, (5, 5), sigma, sigma)
    # apply canny edge detection
    canny = cv.Canny(gaussian, min_edge, min_edge * ratio, 3, L2gradient = True)

    '''
    part of watershed
    '''
    # Finding the contors in the image using chain approximation
    new, contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # converting the marker to float 32 bit
    marker32 = np.int32(marker)
    # Apply watershed algorithm
    cv.watershed(image, marker32)
    # Apply thresholding on the image to convert to binary image
    m = cv.convertScaleAbs(marker32)
    ret, thresh = cv.threshold(m, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Invert the thresh
    thresh_inv = cv.bitwise_not(thresh)
    # Bitwise and with the image mask thresh
    res = cv.bitwise_and(image, image, mask = thresh)
    # Bitwise and the image with mask as threshold invert
    res3 = cv.bitwise_and(image, image, mask = thresh_inv)
    # Take the weighted average
    res4 = cv.addWeighted(res, 1, res3, 1, 0)
    # Draw the contours on the image with green color and pixel width is 1
    final = cv.drawContours(res4, contours, -1, (0, 255, 0), 1)

    # Display the image
    cv.imshow("Watershed", final) 
    # Wait for key stroke
    cv.waitKey()

if __name__ == "__main__":
    print("Hello world")
    #canny_watershed(1, 1, 1, 1)
    #canny_watershed('四破魚(藍圓鰺)2.jpg', 0, 100, 3)
    with open('file_lists.txt', 'r') as f:
        for line in f:
            params = []
            for param in line.split():
                params.append(param)
            canny_watershed(params[0], float(params[1]), int(params[2]), int(params[3]))
    print("end")
