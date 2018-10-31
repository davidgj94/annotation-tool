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

def plot_overlap(image, overlap, alpha=0.5, color=[255,0,0]):
    image = np.float32(image)
    image[overlap] *= 1. - alpha
    image[overlap] += alpha * np.array(color)
    image = np.uint8(image)
    plt.figure()
    plt.imshow(image)
    plt.show()


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

# def blend(image1, image2, ymin, ymax, alpha=0.5):

#     full_image1 = np.zeros((ymax,image1.shape[1],3))
#     full_image2 = np.zeros((ymax,image1.shape[1],3))

#     full_image1[:image1.shape[0],:] = image1
#     full_image1[ymin:,:] = image2

#     full_image2[ymin:,:] = image2
#     full_image2[:image1.shape[0],:] = image1
    
#     full_image = np.uint8(alpha * full_image1 + (1 - alpha) * full_image2)

#     return full_image


def get_offset(shift):
    offset_x = max(0, shift[0])
    offset_y = max(0, shift[1])
    return (offset_x, offset_y)

def paste(new_img, img, offset_x, offset_y):
    h, w, z = img.shape
    new_img[offset_y:offset_y + h, offset_x:offset_x + w] = img
    return new_img

def get_overlap(image1, image2, shift):

    h1, w1, z1 = image1.shape
    h2, w2, z2 = image2.shape

    offset1_x, offset1_y = get_offset(-shift)
    offset2_x, offset2_y = get_offset(shift)
    
    nw, nh = map(max, map(operator.add, (offset1_x, offset1_y), (w1, h1)), map(operator.add, (offset2_x, offset2_y), (w2, h2)))

    mask_1 = paste(np.zeros((nh, nw)), image1, offset1_x, offset1_y)
    mask_2 = paste(np.zeros((nh, nw)), image2, offset2_x, offset2_y)
    overlap = np.logical_and(mask_1, mask_2)

    return overlap


def merge_images(image1, image2, shift, blend=True, alpha=0.5):

    h1, w1, z1 = image1.shape
    h2, w2, z2 = image2.shape

    offset1_x, offset1_y = get_offset(-shift)
    offset2_x, offset2_y = get_offset(shift)
    
    nw, nh = map(max, map(operator.add, (offset1_x, offset1_y), (w1, h1)), map(operator.add, (offset2_x, offset2_y), (w2, h2)))

    new_image = np.zeros((nh, nw, 3))
    new_image = paste(new_image, image1, offset1_x, offset1_y)
    new_image = paste(new_image, image2, offset2_x, offset2_y)

    if blend:

        new_image_aux = np.zeros((nh, nw, 3))
        new_image_aux = paste(new_image_aux, image2, offset2_x, offset2_y)
        new_image_aux = paste(new_image_aux, image1, offset1_x, offset1_y)

        new_image *= alpha
        new_image += (1 - alpha) * new_image_aux
        new_image = np.uint8(new_image)

    return new_image



parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--start_time', type=str, required=True)
parser.add_argument('--end_time', type=str, required=True)
parser.add_argument('--fps', type=float, default=1.0)
args = parser.parse_args()

start_time_msec = string2msec(args.start_time)
end_time_msec = string2msec(args.end_time)
delay_msec = int(1000 * (1 / args.fps))

vidcap = cv2.VideoCapture(args.video_path)

# vidcap.set(cv2.CAP_PROP_POS_MSEC,(end_time_msec))
# success, frame_end = vidcap.read()
# frame_end = downscale(frame_end, 1000, False, pad_img=False)
# mask_end = cv2.imread('end.png')
# mask_end = downscale(mask_end, 1000, True, pad_img=False)

# vidcap.set(cv2.CAP_PROP_POS_MSEC,(start_time_msec))
# success, frame_start = vidcap.read()
# frame_start = downscale(frame_start, 1000, False, pad_img=False)
# mask_start = cv2.imread('start.png')
# mask_start = downscale(mask_start, 1000, True, pad_img=False)

# M = homography(frame_end, frame_start, draw_matches=False)
# total_mask, _ = warp_image(mask_start, M, alpha_channel=False, flags=cv2.INTER_NEAREST)
# total_mask[:mask_end.shape[0],:mask_end.shape[1]] = mask_end
# #full_image, (ymin,ymax) = warp_image(frame_start, M, alpha_channel=False)
# #full_image = blend(frame_end, full_image[ymin:,:], ymin, ymax)
# #plt.figure()
# #plt.imshow(full_image[...,::-1])
# #plt.show()

M = homography(frame_end, frame_start, draw_matches=False)
mask_start_warped, shift = warp_image(mask_start, M, alpha_channel=False, flags=cv2.INTER_NEAREST)
total_mask = merge_images(mask_end, mask_start_warped, blend=False)

#full_image, (ymin,ymax) = warp_image(frame_start, M, alpha_channel=False)
#full_image = blend(frame_end, full_image[ymin:,:], ymin, ymax)
#plt.figure()
#plt.imshow(full_image[...,::-1])
#plt.show()

# current_time = end_time_msec - delay_msec
# success = True
# while success and (current_time > start_time_msec):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,(current_time))
#     success, frame_next = vidcap.read()
#     if success:
#         frame_next = downscale(frame_next, 1000, False, pad_img=False)
#         M = homography(frame_end, frame_next, draw_matches=False)
#         warped, (ymin, ymax) = warp_image(frame_next, M, alpha_channel=False)
#         mask = total_mask[ymin:ymax,:]
#         warped = warped[ymin:ymax,:]
#         mask[mask == 255] = 1
#         vis_img = vis.vis_seg(warped[...,::-1], mask[...,0], vis.make_palette(2))
#         plt.figure()
#         plt.imshow(vis_img)
#         plt.show()
#         print msec2string(current_time)
#         current_time -= delay_msec


current_time = end_time_msec - delay_msec
success = True
while success and (current_time > start_time_msec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(current_time))
    success, frame_next = vidcap.read()
    if success:
        frame_next = downscale(frame_next, 1000, False, pad_img=False)
        M = homography(frame_end, frame_next, draw_matches=False)
        warped, shift = warp_image(frame_next, M, alpha_channel=False)
        h, w, z = warped.shape
        offset_x, offset_y = get_offset(shift)
        mask = total_mask[offset_y:offset_y + h, offset_x:offset_x + w]
        mask[mask == 255] = 1
        vis_img = vis.vis_seg(warped[...,::-1], mask[...,0], vis.make_palette(2))
        plt.figure()
        plt.imshow(vis_img)
        plt.show()
        print msec2string(current_time)
        current_time -= delay_msec


# vidcap.set(cv2.CAP_PROP_POS_MSEC,(end_time_msec))
# success, frame_end = vidcap.read()
# frame_end = downscale(frame_end, 1000, False, pad_img=False)
# current_time = end_time_msec - delay_msec
# while success and (current_time > start_time_msec):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,(current_time))
#     success, frame_next = vidcap.read()
#     if success:
#         frame_next = downscale(frame_next, 1000, False, pad_img=False)
#         M = homography(frame_end, frame_next, draw_matches=False)
#         warped, (ymin, ymax) = warp_image(frame_next, M)
#         frame_end_alpha = np.vstack((
#             np.ones((frame_end.shape[0], frame_end.shape[1])), 
#             np.zeros((warped.shape[0] - frame_end.shape[0], frame_end.shape[1]))
#             ))
#         overlap = np.logical_and(warped[...,-1], frame_end_alpha)
#         warped = cv2.cvtColor(warped, cv2.COLOR_BGRA2BGR)
#         full_image = blend(frame_end, warped[ymin:,:], ymin, ymax)
#         plot_overlap(full_image[...,::-1], overlap)
#         print msec2string(current_time)
#         current_time -= delay_msec

vidcap.set(cv2.CAP_PROP_POS_MSEC,(end_time_msec))
success, frame_end = vidcap.read()
frame_end = downscale(frame_end, 1000, False, pad_img=False)
current_time = end_time_msec - delay_msec
while success and (current_time > start_time_msec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(current_time))
    success, frame_next = vidcap.read()
    if success:
        frame_next = downscale(frame_next, 1000, False, pad_img=False)
        M = homography(frame_end, frame_next, draw_matches=False)
        warped, shift = warp_image(frame_next, M)
        overlap = get_overlap(np.zeros((frame_end.shape[0], frame_end.shape[1])), warped[...,-1], shift)
        warped = cv2.cvtColor(warped, cv2.COLOR_BGRA2BGR)
        full_image = merge_images(frame_end, warped, shift)
        plot_overlap(full_image[...,::-1], overlap)
        print msec2string(current_time)
        current_time -= delay_msec
