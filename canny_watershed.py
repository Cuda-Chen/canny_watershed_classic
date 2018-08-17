# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import os
import random as rng

def canny_watershed(inputfile, outputfile, sigma, min_edge, ratio):
    rng.seed(12345)

    # first, read the image
    #image = cv.imread('coins.jpg')
    #image = cv.imread('四破魚(藍圓鰺)2.jpg')
    image = cv.imread(inputfile)
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

    
    # Create the marker image for watershed algorithm
    markers = np.zeros(canny.shape, dtype = np.int32)
    
    # Draw the foreground markers
    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i + 1) , -1)

    # Draw the background markers
    cv.circle(markers, (5, 5), 3, (255, 255, 255), -1)
    cv.imshow('markers', markers * 10000)

    # Apply watershed algorithm
    cv.watershed(image, markers)
    

    '''
    mark = markers.astype('uint8')
    mark = cv.bitwise_not(mark)
    cv.imshow('marker v2', mark)
    '''
    
    # Apply thresholding on the image to convert to binary image
    m = cv.convertScaleAbs(markers)
    ret, thresh = cv.threshold(m, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow('thresh', thresh)

    # Invert the thresh
    thresh_inv = cv.bitwise_not(thresh)
    cv.imshow('thresh_inv', thresh_inv)

    # Bitwise and with the image mask thresh
    res = cv.bitwise_and(image, image, mask = thresh)
    cv.imshow('res', res)

    # Bitwise and the image with mask as threshold invert
    res3 = cv.bitwise_and(image, image, mask = thresh_inv)
    cv.imshow('res3', res3)
    # Take the weighted average
    res4 = cv.addWeighted(res, 1, res3, 1, 0)
    cv.imshow('marker v2', res4)
    
    
    # Generate random color
    colors = []
    for contour in contours:
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))

    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

    # Fill labeled objects with random color
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i,j]
            if index > 0 and index <= len(contours):
                dst[i,j,:] = colors[index-1]
            #else:
            #    dst[i, j, :] = (0, 0, 0)

    # Display the final result
    cv.imshow('final result', dst)
    
    '''
    # converting the marker to float 32 bit
    marker32 = np.int32(marker)
    # Apply watershed algorithm
    cv.watershed(image, marker32)
    # Apply thresholding on the image to convert to binary image
    m = cv.convertScaleAbs(marker32)
    ret, thresh = cv.threshold(m, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow('thresh', thresh)
    # Invert the thresh
    thresh_inv = cv.bitwise_not(thresh)
    cv.imshow('thresh_inv', thresh_inv)
    # Bitwise and with the image mask thresh
    res = cv.bitwise_and(image, image, mask = thresh)
    cv.imshow('res', res)
    # Bitwise and the image with mask as threshold invert
    res3 = cv.bitwise_and(image, image, mask = thresh_inv)
    cv.imshow('res3', res3)
    # Take the weighted average
    res4 = cv.addWeighted(res, 1, res3, 1, 0)
    cv.imshow('res4', res4)
    # Draw the contours on the image with green color and pixel width is 1
    final = cv.drawContours(res4, contours, -1, (0, 255, 0), 1)
    

    # Display the image
    cv.imshow("Watershed", final) 
    '''
    # Save the output image
    #cv.imwrite(outputfile, final)
    # Wait for key stroke
    cv.waitKey()

if __name__ == "__main__":
    print("Hello world")
    #canny_watershed(1, 1, 1, 1)
    #canny_watershed('四破魚(藍圓鰺)2.jpg', 0, 100, 3)
    #canny_watershed('8ubS9.jpg', 'output.jpg', 0, 100, 3)
    
    with open('file_lists.txt', 'r') as f:
        for line in f:
            params = []
            for param in line.split():
                params.append(param)
            outputfile = os.path.splitext(params[0])[0] + '_' + \
                         params[1] + '_' + params[2] + '_' + params[3] + '.bmp'
            canny_watershed(params[0], outputfile, float(params[1]), int(params[2]), int(params[3]))
            #filename = os.path.splitext(params[0])[0]
            #print(filename)
    
    print("end")
