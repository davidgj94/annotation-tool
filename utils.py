import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import argparse
import time
import pdb
import vis
from downscale import _downscale as downscale
import operator

_FISHESYE_METHOD_DATA = 'datos_calib/fisheye'
_STANDARD_METHOD_DATA = 'datos_calib/standard'

_D = np.load(os.path.join(_FISHESYE_METHOD_DATA,'D.npy'))
_K = np.load(os.path.join(_FISHESYE_METHOD_DATA,'D.npy'))
_dist = np.load(os.path.join(_STANDARD_METHOD_DATA,'dist.npy'))
_mtx = np.load(os.path.join(_STANDARD_METHOD_DATA,'mtx.npy'))


def undistort(img, use_fisheye_method=True, TOTAL=True):

h,  w = img.shape[:2]

if use_fisheye_method:

    if TOTAL:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(_K, _D, (w,h), np.eye(3), balance=1)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(_K, _D, np.eye(3), new_K, (w,h), cv2.CV_16SC2)
    else:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(_K, _D, np.eye(3), K, (w,h), cv2.CV_16SC2)

    dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

else:

    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(_mtx, _dist,(w,h),0,(w,h))
    dst = cv2.undistort(img, _mtx, _dist, None, newcameramtx)

return dst

def homography(image_a, image_b, draw_matches=True):

    image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    kp_a, des_a = sift.detectAndCompute(image_a, None)
    kp_b, des_b = sift.detectAndCompute(image_b, None)

    # Brute force matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, trainDescriptors=des_b, k=2)

    # Lowes Ratio
    good_matches = []
    for m, n in matches:
        if m.distance < .75 * n.distance:
            good_matches.append(m)

    if draw_matches:
        aux_img = cv2.drawMatches(image_a, kp_a, image_b, kp_b, sorted(good_matches, key = lambda x:x.distance)[:10], None, flags=2)
        plt.figure()
        plt.imshow(aux_img),plt.show()

    src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches])\
        .reshape(-1, 1, 2)
    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches])\
        .reshape(-1, 1, 2)

    if len(src_pts) > 4:
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5)
    else:
        M = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    return M



def warp_image(image, homography, alpha_channel=True, is_mask=False):

    if alpha_channel:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    h, w, z = image.shape

    # Find min and max x, y of new image
    p = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    p_prime = np.dot(homography, p)

    yrow = p_prime[1] / p_prime[2]
    xrow = p_prime[0] / p_prime[2]
    xmin = min(xrow)
    ymax = max(yrow)
    ymin = min(yrow)
    xmax = max(xrow)

    new_mat = np.array([[1, 0, -1 * xmin], [0, 1, -1 * ymin], [0, 0, 1]])
    homography = np.dot(new_mat, homography)

    width = int(round(xmax - xmin))
    heigth = int(round(ymax - ymin))

    size = (width, heigth)
    if is_mask:
        warped = cv2.warpPerspective(src=image, M=homography, dsize=size, flags=cv2.INTER_NEAREST)
    else:
        warped = cv2.warpPerspective(src=image, M=homography, dsize=size, flags=cv2.INTER_LINEAR)

    shift = (int(xmin), int(ymin))

    return warped, shift


def string2msec(time_string):
    time_min = int(time_string.split(':')[0])
    time_sec = int(time_string.split(':')[1])
    time_sec += time_min * 60
    time_msec = 1000 * time_sec
    return time_msec



def msec2string(time_msec):
    time_sec = time_msec / 1000
    time_min = time_sec / 60
    time_string = "{}:{:02d}".format(time_min, time_sec - time_min * 60)
    return time_string



def get_offset(shift):
    offset_x = max(0, shift[0])
    offset_y = max(0, shift[1])
    return (offset_x, offset_y)



def paste(new_img, img, offset):
    h, w, z = img.shape
    offset_x, offset_y = offset
    new_img[offset_y:offset_y + h, offset_x:offset_x + w] = img
    return new_img



def merge_images(image1, image2, shift, blend=True, alpha=0.5):

    h1, w1, z1 = image1.shape
    h2, w2, z2 = image2.shape

    offset1 = get_offset((-shift[0], -shift[1]))
    offset2 = get_offset(shift)
    
    nw, nh = map(max, map(operator.add, offset1, (w1, h1)), map(operator.add, offset2, (w2, h2)))

    new_image = np.zeros((nh, nw, 3))
    new_image = paste(new_image, image1, offset1)
    new_image = paste(new_image, image2, offset2)

    if blend:

        new_image_aux = np.zeros((nh, nw, 3))
        new_image_aux = paste(new_image_aux, image2, offset2)
        new_image_aux = paste(new_image_aux, image1, offset1)

        new_image *= alpha
        new_image += (1 - alpha) * new_image_aux
        new_image = np.uint8(new_image)

    return new_image, offset1, offset2