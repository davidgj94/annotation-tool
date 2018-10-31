import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import argparse
import time
import pdb
import vis

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

def warp_image(image, homography, alpha_channel=True, flags=cv2.INTER_LINEAR):

    if alpha_channel:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    h, w, z = image.shape

    # Find min and max x, y of new image
    p = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    p_prime = np.dot(homography, p)

    yrow = p_prime[1] / p_prime[2]
    xrow = p_prime[0] / p_prime[2]
    #xmin = min(xrow)
    ymax = max(yrow)
    ymin = min(yrow)
    #xmax = max(xrow)

    #width = int(round(xmax - xmin))
    #heigth = int(round(ymax - ymin))

    size = (4096, int(ymax))

    warped = cv2.warpPerspective(src=image, M=homography, dsize=size, flags=flags)

    # if xmin < 0:
    #     warped = warped[:,int(-xmin):,:]
    # if xmin > 0:
    #     warped = cv2.copyMakeBorder(warped, top=0, bottom=0, left=int(xmax), right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0,0])
    # if xmax > 4096:
    #     pdb.set_trace()
    #     warped = warped[:,:-(warped.shape[1] - 4096),:]
    # if warped.shape[1] < 4096:
    #     warped = cv2.copyMakeBorder(warped, top=0, bottom=0, left=0, right=(4096 - warped.shape[1]), borderType= cv2.BORDER_CONSTANT, value=[0,0,0,0])

    return warped, (int(ymin), int(ymax))

def plot_overlap(alpha1, alpha2):
    overlap = np.logical_and(alpha1, alpha2)
    plt.figure()
    plt.imshow(overlap)
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

def blend(image1, image2, ymin, ymax, alpha=0.5):

    full_image1 = np.zeros((ymax,4096,3))
    full_image2 = np.zeros((ymax,4096,3))

    full_image1[:image1.shape[0],:] = image1
    full_image1[ymin:,:] = image2

    full_image2[ymin:,:] = image2
    full_image2[:image1.shape[0],:] = image1
    
    full_image = np.uint8(alpha * full_image1 + (1 - alpha) * full_image2)

    return full_image


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
# current_time = end_time_msec - delay_msec
# while success and (current_time > start_time_msec):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,(current_time))
#     success, frame_next = vidcap.read()
#     if success:
#         M = homography(frame_end, frame_next, draw_matches=False)
#         warped = warp_image(frame_next, M)
#         frame_end_alpha = np.vstack((
#             np.ones((frame_end.shape[0], frame_end.shape[1])), 
#             np.zeros((warped.shape[0] - frame_end.shape[0], frame_end.shape[1]))
#             ))
#         plot_overlap(warped[:,:,-1], frame_end_alpha)
#         print msec2string(current_time)
#         current_time -= delay_msec


# vidcap.set(cv2.CAP_PROP_POS_MSEC,(end_time_msec))
# success, frame_end = vidcap.read()
# cv2.imwrite('end.png', frame_end)

# vidcap.set(cv2.CAP_PROP_POS_MSEC,(start_time_msec))
# success, frame_start = vidcap.read()
# cv2.imwrite('start.png', frame_start)


vidcap.set(cv2.CAP_PROP_POS_MSEC,(end_time_msec))
success, frame_end = vidcap.read()
mask_end = cv2.imread('end.png')

vidcap.set(cv2.CAP_PROP_POS_MSEC,(start_time_msec))
success, frame_start = vidcap.read()
mask_start = cv2.imread('start.png')

M = homography(frame_end, frame_start, draw_matches=False)
total_mask, _ = warp_image(mask_start, M, alpha_channel=False, flags=cv2.INTER_NEAREST)
total_mask[:mask_end.shape[0],:mask_end.shape[1]] = mask_end
full_image, (ymin,ymax) = warp_image(frame_start, M, alpha_channel=False)
#full_image[:frame_end.shape[0],:frame_end.shape[1]] = frame_end
# frame_end_alpha = np.vstack((
#             np.ones((frame_end.shape[0], frame_end.shape[1])), 
#             np.zeros((full_image.shape[0] - frame_end.shape[0], frame_end.shape[1]))
#             ))
# overlap = np.logical_and(full_image[...,-1], frame_end_alpha)
full_image = blend(frame_end, full_image[ymin:,:], ymin, ymax)
# plt.figure()
# plt.imshow(full_image[...,::-1])
# plt.show()


vidcap.set(cv2.CAP_PROP_POS_MSEC,(end_time_msec))
success, frame_end = vidcap.read()
current_time = end_time_msec - delay_msec
while success and (current_time > start_time_msec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(current_time))
    success, frame_next = vidcap.read()
    if success:
        M = homography(frame_end, frame_next, draw_matches=False)
        warped, (ymin, ymax) = warp_image(frame_next, M, alpha_channel=False)
        mask = total_mask[ymin:ymax,:]
        warped = warped[ymin:ymax,:]
        mask[mask == 255] = 1
        vis_img = vis.vis_seg(warped[...,::-1], mask[...,0], vis.make_palette(2))
        plt.figure()
        plt.imshow(vis_img)
        plt.show()
        print msec2string(current_time)
        current_time -= delay_msec



