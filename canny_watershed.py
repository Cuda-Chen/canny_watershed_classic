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
    part of mean shift
    '''
    # convert image from unsigned 8 bit to 32 bit float
    image_float = np.float32(image)

    # define the criteria(type, max_iter, epsilon)
    # cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
    # cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
    # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
    # max_iter - An integer specifying maximum number of iterations.In this case it is 10
    # epsilon - Required accuracy.In this case it is 1
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)

    k = 50 # number of clusters
    #apply K-means algorithm with random centers approach
    ret, label, centers = cv.kmeans(image_float, k, None, criteria, 50, cv.KMEANS_RANDOM_CENTERS)

    # convert the image from 32 bit float to unsigned 8 bit
    center = np.uint8(centers)
    # this will flatten the label
    res = center[label.flatten()]
    # reshape the image
    res2 = res.reshape(image.shape)
    cv.imshow('K means', res2)

    # apply meanshift algorithm on to image
    meanshift = cv.pyrMeanShiftFiltering(image, sp = 8, sr = 16, maxLevel = 1, termcrit = (cv.TERM_CRITERIA_EPS+ cv.TERM_CRITERIA_MAX_ITER, 5, 1))
    cv.imshow('mean shift', meanshift)

    '''
    part of misc
    '''
    # change image from BGR to grayscale
    #gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(meanshift, cv.COLOR_BGR2GRAY)
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
    part of watershed
    '''
    # Finding the contors in the image using chain approximation
    #new, contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    new, contours, hierarchy = cv.findContours(marker, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    
    # Create the marker image for watershed algorithm
    #markers = np.zeros(canny.shape, dtype = np.int32)
    markers = np.zeros(marker.shape, dtype = np.int32)
    
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
    #for i in range(len(contours)):
    #    cv.drawContours(res4, contours, i, (i + 1), -1)
    

    # Display the image
    #cv.imshow("Watershed", final) 
    cv.imshow("Watershed", res4)
    '''
    '''
    colors = []
    for contour in contours:
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))

    # Create the result image
    dst = np.zeros((marker32.shape[0], marker32.shape[1], 3), dtype=np.uint8)

    # Fill labeled objects with random color
    for i in range(marker32.shape[0]):
        for j in range(marker32.shape[1]):
            index = marker32[i,j]
            if index > 0 and index <= len(contours):
                dst[i,j,:] = colors[index-1]
    cv.imshow('final', dst)
    '''
    # Save the output image
    #cv.imwrite(outputfile, final)
    # Wait for key stroke
    cv.waitKey()

def canny_watershed_distance_transform(inputfile, outputfile, sigma, min_edge, ratio):
    # read image -> convert to grayscale -> apply Otus thresholding
    img = cv.imread(inputfile)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow('otsu', thresh)
    

    # noise removal (using Canny)
    gaussian = cv.GaussianBlur(thresh, (5, 5), sigma, sigma)
    canny = cv.Canny(gaussian, min_edge, min_edge * ratio, 3, L2gradient = True)

    # sharpen the image
    mask = canny != 0
    imgResult = img * (mask[:,:,None].astype(img.dtype))
    #sharp = np.uint8(img)
    #imgResult = sharp - canny

    # convert back to 8bits gray scale
    #imgResult = np.clip(imgResult, 0, 255)
    #imgResult = imgResult.astype('uint8')

    cv.imshow('canny', canny)
    cv.imshow('new sharpen image', imgResult)

    # need Otsu thresholding again?

    # apply distance transform
    dist = cv.distanceTransform(canny, cv.DIST_L2, 3)
    #print(canny.dtype)
    #print(dist.dtype)
    
    # normalize the distance image for range {0.0, 1.0}
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow('distance transform image', dist)

    # Threshold to obtain the peaks
    # This will be the markers for the foreground objects
    ret, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)

    # Dilate a bit the dist image
    kernel1 = np.ones((3,3), dtype=np.uint8)
    dist = cv.dilate(dist, kernel1)
    cv.imshow('Peaks', dist)

    # Create the CV_8U version of the distance image
    # It is needed for findContours()
    dist_8u = dist.astype('uint8')
    # Find total markers
    ret, contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i+1), -1)
    # Draw the background marker
    cv.circle(markers, (5,5), 3, (255,255,255), -1)
    cv.imshow('Markers', markers*10000)

    # Perform the watershed algorithm
    cv.watershed(imgResult, markers)
    #mark = np.zeros(markers.shape, dtype=np.uint8)
    mark = markers.astype('uint8')
    mark = cv.bitwise_not(mark)
    # uncomment this if you want to see how the mark
    # image looks like at that point
    cv.imshow('Markers_v2', mark)
    # Generate random colors
    colors = []
    for contour in contours:
        colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i,j]
            if index > 0 and index <= len(contours):
                dst[i,j,:] = colors[index-1]
    # Visualize the final image
    cv.imshow('Final Result', dst)

    cv.waitKey()

if __name__ == "__main__":
    print("Hello world")
    #canny_watershed(1, 1, 1, 1)
    #canny_watershed('四破魚(藍圓鰺)2.jpg', 'output.jpg', 0, 100, 3)
    #canny_watershed('coins.jpg', 'output.jpg', 0, 100, 3)
    canny_watershed_distance_transform('四破魚(藍圓鰺)2.jpg', 'output.jpg', 0, 100, 3)
    #canny_watershed('七星鱸.JPG', 'output.jpg', 0, 100, 3)
    '''
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
    '''
    print("end")
