import utils
from utils import homography, warp_image, string2msec, msec2string, merge_images, undistort
import os.path
import os
import json
import cv2
from Tkinter import * 
from matplotlib import pyplot as plt
import argparse
import numpy as np
from downscale import _downscale as downscale


def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_path', type=str, required=True)
	parser.add_argument('--start_time', type=str, required=True)
	parser.add_argument('--end_time', type=str, required=True)
	parser.add_argument('--fps', type=float, default=1.0)
	parser.add_argument('--nframes', type=int, default=2)
	return parser

def preprocess_frame(frame):
	frame = undistort(frame)
	frame = downscale(frame, 1000, False, pad_img=False)
	return frame


args = make_parser().parse_args()

start_time_msec = string2msec(args.start_time)
end_time_msec = string2msec(args.end_time)
delay_msec = int(1000 * (args.nframes / args.fps))

vidcap = cv2.VideoCapture(video_path)
vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time_msec)
success, start_frame = vidcap.read()
if success:
	start_frame = preprocess_frame(start_frame)


M_list = []
frames_list = [start_frame]
current_time = start_time_msec + delay_msec

while current_time < end_time_msec:
	vidcap.set(cv2.CAP_PROP_POS_MSEC, current_time)
	success, current_frame = vidcap.read()
	if success:
		current_frame = preprocess_frame(current_frame)
		frames_list.append(current_frame)
		M = homography(frames_list[-2], frames_list[-1], draw_matches=False)
		M_list.append(M)
	current_time += delay_msec


_M = M_list[0]
warped, offset = warp_image(frames_list[1], _M)
warped_list = [warped]
offset_list = [offset]

for i in range(2,len(frames_list)):
	_M = np.dot(M_list[i-1], _M)
	warped, offset = warp_image(frames_list[i], _M)
	warped_list.append(warped)
	offset_list.append(offset)

warped_list.insert(0, cv2.cvtColor(frames_list[0], cv2.COLOR_BGR2BGRA))
offset_list = insert(0,(0,0))

offset_x_list = [offset_x for offset_x, _ in offset_list]
xmax_list = []
for offset_x, img in zip(offset_x_list, warped_list):
	xmax_list.append(offset_x + img.shape[1])


offset_y_list = [offset_y for _, offset_y in offset_list]
ymax_list = []
for offset_y, img in zip(offset_y_list, warped_list):
	ymax_list.append(offset_y + img.shape[1])


xmin = min(offset_x_list)
xmax = max(xmax_list)
W = xmax - xmin

ymin = min(offset_y_list)
ymax = max(ymax_list)
H = ymax - ymin

offset_list = [(offset_x - xmin, offset_y - ymin) for offset_x, offset_y in offset_list]

alpha_channels_list = []
for warped, offset in zip(warped_list, offset_list):
	alpha_channel = np.zeros((H, W), dtype=bool)
	y, x = np.where(warped[...,-1])
	coords = np.array(zip(x,y))
	coords += np.array(offset)
	r = coords[:,1].flat()
	c = coords[:,0].flat()
	alpha_channel[r,c] = True
	alpha_channels_list.append(alpha_channels)

overlaps_list = []

for i in range(1,len(alpha_channels)-1):
	overlap1 = np.logical_and(alpha_channels[i], alpha_channels[i-1])
	overlap2 = np.logical_and(alpha_channels[i], alpha_channels[i+1])
	overlap3 = np.logical_and(overlap1, overlap2)
	overlap1 = np.logical_and(overlap1, ~overlap3)
	overlap2 = np.logical_and(overlap2, ~overlap3)
	overlaps_list.append((overlap1, overlap2, overlap3))


pano = np.zeros((H, W, 3), dtype=np.uint8)
for overlap1, overlap2, overlap3 in overlaps_list:
	pano










