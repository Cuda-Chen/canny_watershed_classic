# -*- coding: UTF-8 -*-
#import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv
import random as rng
import argparse
import time

#import canny_watershed as cannyWshed

from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

def mean_shift(inputfile, sp, sr):
    rng.seed(12345)

    image = cv.imread(inputfile)
    print("the shape of image is ", image.shape)
    print("thie dtype of image is ", image.dtype)

    '''create mask image'''
    mask_image = np.zeros(image.shape, dtype=float)
    

    '''part of mean shift'''
    meanshift = cv.pyrMeanShiftFiltering(image, sp, sr, maxLevel=1, termcrit=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 5, 1))

    '''
    part of misc
    '''
    # change image from BGR to grayscale
    #gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(meanshift, cv.COLOR_BGR2GRAY)
    # apply thresholding to convert the image to binary
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # erode the image
    foreground = cv.erode(thresh, None, iterations=1)
    # Dilate the image
    backgroundTemp = cv.dilate(thresh, None, iterations=1)
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
    markers = np.zeros(marker.shape, dtype=np.int32)

    # Draw the foreground markers
    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i + 1) , -1)

    # Draw the background markers
    cv.circle(markers, (5, 5), 3, (255, 255, 255), -1)
    #cv.imshow('markers', markers * 10000)

    # Apply watershed algorithm
    cv.watershed(image, markers)

    # Apply thresholding on the image to convert to binary image
    m = cv.convertScaleAbs(markers)
    ret, thresh = cv.threshold(m, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #cv.imshow('thresh', thresh)

    # Invert the thresh
    thresh_inv = cv.bitwise_not(thresh)
    #cv.imshow('thresh_inv', thresh_inv)

    # Bitwise and with the image mask thresh
    res = cv.bitwise_and(image, image, mask=thresh)
    #cv.imshow('res', res)

    # Bitwise and the image with mask as threshold invert
    res3 = cv.bitwise_and(image, image, mask=thresh_inv)
    #cv.imshow('res3', res3)
    # Take the weighted average
    res4 = cv.addWeighted(res, 1, res3, 1, 0)
    #cv.imshow('marker v2', res4)

    '''
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
    '''

    # draw the contours on the image with red color and pixel width is 1
    #final = cv.drawContours(res4, contours, -1, (255, 0, 0), 1)
    final = cv.drawContours(mask_image, contours, -1, (255, 0, 0), 1)
    #print("the shape of contours is ", contours.shape)
    #print("the dtype of contours is ", contours.dtype)

    #cv.imshow('edge map', contours)
    #cv.waitKey(0)

    #return dst
    return final

def cannyWatershed(inputfile):
    img = io.imread(inputfile)
    #img = cv.imread(inputfile)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    high_thresh, thresh_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    low_thresh = high_thresh * 0.5 
    marker = cv.GaussianBlur(gray, (5, 5), 2)
    #canny = cv.Canny(marker, 40, 100)
    canny = cv.Canny(marker, low_thresh, high_thresh)
    _, contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    marker32 = np.int32(marker)
    watershed = cv.watershed(img, marker32)
    m = cv.convertScaleAbs(marker32)
    _, thresh = cv.threshold(m, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    thresh_inv = cv.bitwise_not(thresh)
    temp = cv.bitwise_and(img, img, mask=thresh)
    temp1 = cv.bitwise_and(img, img, mask=thresh_inv)
    result = cv.addWeighted(temp, 1, temp1, 1, 0)
    final = cv.drawContours(result, contours, -1, (0, 0, 255), 1)
    mask = np.zeros(img.shape, dtype=float)
    edgemap = cv.drawContours(mask, contours, -1, (255, 0, 0), 1)

    return edgemap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='input image path')
    parser.add_argument('meanshift_sp', type=int, help='mean shift sp')
    parser.add_argument('meanshift_sr', type=int, help='mean shift sr')
    parser.add_argument('fz_min_size', type=int, help='felzenswalb segment min size')
    parser.add_argument('SLIC_n_segments', type=int, help='SLIC minimum number of segment')
    parser.add_argument('quick_shift_max_dist', type=int, help='quickshift max iter')
    args = parser.parse_args()

    print("hello")

    inputfile = args.input_file
    superpixel_sigma = 0.5
    superpixel_color = (1, 0, 0)

    output_meanshift = os.path.splitext(inputfile)[0] + '_meanshift_' + str(args.meanshift_sp) + '_' + str(args.meanshift_sr) + '.bmp'
    output_felzenszwalb = os.path.splitext(inputfile)[0] + '_felzenswalb_' + str(args.fz_min_size) + '.bmp'
    output_slic = os.path.splitext(inputfile)[0] + '_slic_' + str(args.SLIC_n_segments) + '.bmp'
    output_quickshift = os.path.splitext(inputfile)[0] + '_quickshift_' + str(args.quick_shift_max_dist) + '.bmp'
    output_result = os.path.splitext(inputfile)[0] + '_result' + '.bmp'

    print(output_meanshift)
    print(output_felzenszwalb)
    print(output_slic)
    print(output_quickshift)

    # mean shift
    tStart = time.time()
    #meanshift_result = mean_shift(inputfile, sp=args.meanshift_sp, sr=args.meanshift_sr)
    #meanshift_result = cannyWshed.cannyWatershed(inputfile)
    meanshift_result = cannyWatershed(inputfile)
    print("the shape of meanshift_result is ", meanshift_result.shape)
    print("the dtype of meanshift_result is ", meanshift_result.dtype)
    tEnd = time.time()
    print("Mean shift cost %f sec" % (tEnd - tStart))

    #image = img_as_float(io.imread(inputfile))
    image = io.imread(inputfile)
    #mask_img = np.zeros(image.shape, dtype=np.uint8)
    mask_img = np.zeros(image.shape, dtype=float)

    # felzenszwalb
    tStart = time.time()
    segment_felzenszwalb = felzenszwalb(image, sigma=superpixel_sigma, min_size=args.fz_min_size)
    print("the shape of segment_felzenszwalb is ", segment_felzenszwalb.shape)
    print("the dtype of segment_felzenszwalb is ", segment_felzenszwalb.dtype)
    tEnd = time.time()
    print("felzenszwalb_result cost %f sec" % (tEnd - tStart))

    # SLIC
    tStart = time.time()
    segment_slic = slic(image, sigma=superpixel_sigma, n_segments=args.SLIC_n_segments)
    print("the shape of segment_slic is ", segment_slic.shape)
    print("the dtype of segment_slic is ", segment_slic.dtype)
    tEnd = time.time()
    print("SLIC cost %f sec" % (tEnd - tStart))

    # quickshift
    tStart = time.time()
    segment_quickshift = quickshift(image, kernel_size=5, max_dist=args.quick_shift_max_dist, ratio=0.5)
    print("the shape of segment_quickshift is ", segment_quickshift.shape)
    print("the dtype of segment_quickshift is ", segment_quickshift.dtype)
    tEnd = time.time()
    print("quickshift cost %f sec" % (tEnd - tStart))

    felzenszwalb_result = mark_boundaries(mask_img, segment_felzenszwalb, color=superpixel_color)
    slic_result = mark_boundaries(mask_img, segment_slic, color=superpixel_color)
    quickshift_result = mark_boundaries(mask_img, segment_quickshift, color=superpixel_color)
    print("the shape of felzenszwalb_result is ", felzenszwalb_result.shape)
    print("the dtype of felzenszwalb_result is ", felzenszwalb_result.dtype)
    print("the shape of slic_result is ", slic_result.shape)
    print("the dtype of slic_result is ", slic_result.dtype)
    print("the shape of quickshift_result is ", quickshift_result.shape)
    print("the dtype of quickshift_result is ", quickshift_result.dtype)

    '''
    fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(meanshift_result)
    ax[0, 0].set_title('mean shift')
    ax[0, 1].imshow(felzenszwalb_result)
    ax[0, 1].set_title('felzenszwalb')
    ax[1, 0].imshow(slic_result)
    ax[1, 0].set_title('SLIC')
    ax[1, 1].imshow(quickshift_result)
    ax[1, 1].set_title('quickshift')

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
    '''
    
    meanshift_result_temp = meanshift_result.astype(np.uint8)
    #cv.imwrite(output_meanshift, meanshift_result)
    io.imsave(output_meanshift, meanshift_result_temp)
    io.imsave(output_felzenszwalb, felzenszwalb_result)
    io.imsave(output_slic, slic_result)
    io.imsave(output_quickshift, quickshift_result)
    
    
    #cv.imshow("mean shift", meanshift_result)
    #cv.imshow("felzenszwalb", felzenszwalb_result)
    #cv.imshow("slic", slic_result)
    #cv.imshow("quick shift", quickshift_result)
    #cv.waitKey(0)
    
    temp_or = cv.bitwise_or(meanshift_result, felzenszwalb_result) 
    temp_or = cv.bitwise_or(temp_or, slic_result)
    temp_or = cv.bitwise_or(temp_or, quickshift_result)
    temp_xor = cv.bitwise_xor(meanshift_result, felzenszwalb_result) 
    temp_xor = cv.bitwise_xor(felzenszwalb_result, slic_result)
    temp_xor = cv.bitwise_xor(temp_xor, quickshift_result)
    temp_and = cv.bitwise_and(felzenszwalb_result, slic_result)

    temp_xor8 = temp_xor.astype(np.uint8)
    temp_xor8 = temp_xor8 * 255
    img2gray = cv.cvtColor(temp_xor8, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    temp = cv.bitwise_and(temp_or, temp_or, mask=mask_inv)
    #temp = cv.bitwise_and(image, temp)
    #cv.imshow("temp or result", temp_or)
    #cv.imshow("temp xor result", temp_xor)
    #cv.imshow("temp and result", temp_and)
    #cv.imshow("temp xor 8bit", temp_xor8)
    #cv.imshow("temp gray", img2gray)
    #cv.imshow("mask", mask)
    #cv.imshow("mask inv", mask_inv)
    cv.imshow("temp result", temp)
    cv.waitKey(0)
    
    #cv.imwrite(output_result, temp)
    io.imsave(output_result, temp)
