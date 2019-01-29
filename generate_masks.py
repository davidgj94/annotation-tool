import utils
from utils import homography, warp_image, string2msec, msec2string, merge_images, get_offset, undistort
import utils
import json
import cv2
import os.path
import vis
import argparse
from downscale import _downscale as downscale
import pdb
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import PIL.Image
import PIL.ImageDraw
from math import ceil
from matplotlib.patches import Circle

def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--work_dir', type=str, required=True)
	parser.add_argument('--labels', type=str, default="labels_mapping.txt")
	return parser

def check_point(point, sz):

	h, w = sz

	if point[0] >= w:
		x = w - 1
	elif point[0] < 0:
		x = 0
	else:
		x = point[0]


	if point[1] >= h:
		y = h - 1
	elif h < 0:
		y = 0
	else:
		y = point[1]

	return (x, y)
    


def json2mask(annotation_path, labels_mapping, sz=(2160, 4096)):

	mask = np.zeros(sz).astype(np.uint8)
	test_mask = np.zeros(sz).astype(np.uint8)

	mask = PIL.Image.fromarray(mask)
	test_mask = PIL.Image.fromarray(test_mask)

	draw = PIL.ImageDraw.Draw(mask)
	test_draw = PIL.ImageDraw.Draw(test_mask)

	with open(annotation_path, 'r') as f:
		json_data = json.load(f)

	for shape in json_data["shapes"]:

		npoints = len(shape["points"])
		xy = [check_point(tuple(point), sz) for point in shape["points"]]
		label_mapping = labels_mapping[shape["label"]]

		if npoints == 2:
			draw.line(xy=xy, fill=label_mapping, width=16)
			test_draw.line(xy=xy, fill=255, width=16)
		elif npoints > 2:
			draw.polygon(xy=xy, fill=label_mapping)
			test_draw.polygon(xy=xy, fill=label_mapping)

	for shape in json_data["shapes"]:

		npoints = len(shape["points"])
		xy = [check_point(tuple(point), sz) for point in shape["points"]]
		label_mapping = labels_mapping[shape["label"]]

		if npoints == 2:
			test_draw.line(xy=xy, fill=label_mapping, width=2)

	concat_mask = np.dstack((np.array(mask), np.array(test_mask)))

	return concat_mask

def select_section(sections, section_name):

	section_start = section_name.split("-")[0]
	section_end = section_name.split("-")[1]

	for section in sections:
		if (section['start'] == section_start) and (section['end'] == section_end):
			return section
	return None


def preprocess_img(img):

	img = undistort(img)
	# img_bgr_down = downscale(img[...,:-1], 1000, False, pad_img=False)
	# alpha_channel_down = downscale(img[...,-1], 1000, True, pad_img=False)
	# new_img = np.dstack((img_bgr_down, alpha_channel_down))
	# return new_img
	img = downscale(img, 1000, False, pad_img=False)
	return img

def fix_mask(warped, mask, sz):

	#warped = warped[:mask.shape[0], :mask.shape[1]]
	y, x = np.where(warped[...,-1])
	pts = np.array(zip(x,y))

	rect = np.zeros((4, 2), dtype = np.float32)

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	H, W = sz[:2]
	pts_dst = np.float32([[0,0],[W,0],[W,H],[0,H]])

	M = cv2.getPerspectiveTransform(rect, pts_dst)
	new_mask = cv2.warpPerspective(mask.copy(), M, (W, H), flags=cv2.INTER_NEAREST)

	return new_mask



def pad_imgs(mask_dir, img_dir, mask_test_dir):

	h_list = []
	w_list = []
	for glob in Path(mask_dir).glob("*.png"):

		img_name = glob.parts[-1]
		mask = cv2.imread(os.path.join(mask_dir, img_name))
		h, w = mask.shape[:2]
		h_list.append(h)
		w_list.append(w)

	h_max = max(h_list)
	w_max = max(w_list)

	H = h_max + int(ceil(float(h_max) / 32) * 32 - h_max)
	W = w_max + int(ceil(float(w_max) / 32) * 32 - w_max)


	for glob in Path(mask_dir).glob("*.png"):

		img_name = glob.parts[-1]
		mask = cv2.imread(os.path.join(mask_dir, img_name))
		img = cv2.imread(os.path.join(img_dir, img_name))
		mask_test = cv2.imread(os.path.join(mask_test_dir, img_name))

		img_padded = utils.pad_img(img, (H, W))
		mask_padded = utils.pad_img(mask, (H,W))
		mask_test_padded = utils.pad_img(mask_test, (H,W))

		cv2.imwrite(os.path.join(mask_dir, img_name), mask_padded)
		cv2.imwrite(os.path.join(img_dir, img_name), img_padded)
		cv2.imwrite(os.path.join(mask_test_dir, img_name), mask_test_padded)

	print (H, W)

def generate_masks_section(vidcap, fps, section, save_dir, vis_dir, img_dir, save_test_dir, full_mask):


	start_time_msec = string2msec(section["start"])
	end_time_msec = string2msec(section["end"])
	offset_start = tuple(section["offset_start"])
	offset_end = tuple(section["offset_end"])
	delay_msec = int(1000 * (1 / fps))

	vidcap.set(cv2.CAP_PROP_POS_MSEC,(start_time_msec))
	success, frame_start = vidcap.read()
	frame_start = preprocess_img(frame_start)

	vidcap.set(cv2.CAP_PROP_POS_MSEC,(end_time_msec))
	success, frame_end = vidcap.read()
	frame_end = preprocess_img(frame_end)

	M = homography(frame_start, frame_end, draw_matches=False)
	warped_end, _ = warp_image(frame_end, M)

	offset_start_x, offset_start_y = offset_start
	offset_end_x, offset_end_y = offset_end
	
	#mask_start = full_mask[offset_start_y:offset_start_y + frame_start.shape[0], offset_start_x:offset_start_x + frame_start.shape[1]]
	mask_start, mask_start_test = np.dsplit(full_mask[offset_start_y:offset_start_y + frame_start.shape[0], offset_start_x:offset_start_x + frame_start.shape[1]], 2)
	mask_start = np.squeeze(mask_start)
	mask_start_test = np.squeeze(mask_start_test)
	vis_start = vis.vis_seg(frame_start, mask_start, vis.make_palette(3))

	cv2.imwrite(os.path.join(save_dir, "{}.png".format(section["start"])), mask_start)
	cv2.imwrite(os.path.join(save_test_dir, "{}.png".format(section["start"])), mask_start_test)
	cv2.imwrite(os.path.join(vis_dir, "{}.png".format(section["start"])), vis_start)
	cv2.imwrite(os.path.join(img_dir, "{}.png".format(section["start"])), frame_start)
	
	mask_end_ = full_mask[offset_end_y:offset_end_y + warped_end.shape[0], offset_end_x:offset_end_x + warped_end.shape[1]]
	#mask_end, mask_end_test = np.dsplit(full_mask[offset_end_y:offset_end_y + warped_end.shape[0], offset_end_x:offset_end_x + warped_end.shape[1]], 2)
	mask_end_ = fix_mask(warped_end, mask_end_, frame_end.shape)
	mask_end, mask_end_test = np.dsplit(mask_end_, 2)
	mask_end = np.squeeze(mask_end)
	mask_end_test = np.squeeze(mask_end_test)
	vis_end = vis.vis_seg(frame_end, mask_end, vis.make_palette(3))

	cv2.imwrite(os.path.join(save_dir, "{}.png".format(section["end"])), mask_end)
	cv2.imwrite(os.path.join(save_test_dir, "{}.png".format(section["end"])), mask_end_test)
	cv2.imwrite(os.path.join(vis_dir, "{}.png".format(section["end"])), vis_end)
	cv2.imwrite(os.path.join(img_dir, "{}.png".format(section["end"])), frame_end)

	success = True
	current_time = start_time_msec + delay_msec
	while success and (current_time < end_time_msec):
	    vidcap.set(cv2.CAP_PROP_POS_MSEC,(current_time))
	    success, frame_next = vidcap.read()
	    if success:

	    	frame_next = preprocess_img(frame_next)
	        M = homography(frame_start, frame_next, draw_matches=False)
	        warped, shift = warp_image(frame_next, M)

	        h, w, z = warped.shape
	        offset_x, offset_y = shift
	        offset_x += offset_start_x
	        offset_y += offset_start_y
	        if offset_x < 0 or offset_y <0:
	        	print (offset_x, offset_y)
	        	print msec2string(current_time)
	        	print
	        offset_x, offset_y = get_offset((offset_x, offset_y))

	        mask_ = full_mask[offset_y:offset_y + h, offset_x:offset_x + w]
	        #mask, mask_test = np.dsplit(full_mask[offset_y:offset_y + h, offset_x:offset_x + w], 2)
	        mask_ = fix_mask(warped, mask_, frame_next.shape)
	        mask, mask_test = np.dsplit(mask_, 2)
	        mask = np.squeeze(mask)
	        mask_test = np.squeeze(mask_test)
	        vis_img = vis.vis_seg(frame_next, mask, vis.make_palette(3))

	        cv2.imwrite(os.path.join(save_dir, "{}.png".format(msec2string(current_time))), mask)
	        cv2.imwrite(os.path.join(save_test_dir, "{}.png".format(msec2string(current_time))), mask_test)
	        cv2.imwrite(os.path.join(vis_dir, "{}.png".format(msec2string(current_time))), vis_img)
	        cv2.imwrite(os.path.join(img_dir, "{}.png".format(msec2string(current_time))), frame_next)

	    current_time += delay_msec



if __name__ == "__main__":

	args = make_parser().parse_args()

	annotations_dir = os.path.join(args.work_dir, "annotations")
	save_dir = os.path.join(args.work_dir, "masks")
	save_test_dir = os.path.join(args.work_dir, "masks_test")
	vis_dir = os.path.join(args.work_dir, "vis")
	img_dir = os.path.join(args.work_dir, "images")
	json_path = os.path.join(args.work_dir, "sections.json")

	with open(json_path, 'r') as f:
		json_data = json.load(f)

	fps = float(json_data['fps'])
	video_path = json_data['video_path']
	sections = json_data['sections']
	merged_dir = json_data['save_dir']

	with open(args.labels, 'r') as f:
		labels_mapping = dict()
		for line in f:
			line = line.replace(" ", "")
			label_name = line.split(':')[0]
			label_integer = int(line.split(':')[1])
			labels_mapping[label_name] = label_integer

	vidcap = cv2.VideoCapture(video_path)

	for glob in Path(annotations_dir).glob('*.json'):

		section_name = os.path.splitext(os.path.basename(glob.parts[-1]))[0]

		img_section = cv2.imread(os.path.join(merged_dir, section_name + '.png'))
		full_mask = json2mask(os.path.join(annotations_dir, glob.parts[-1]), labels_mapping, sz=img_section.shape[:2])
		
		section = select_section(sections, section_name)
		generate_masks_section(vidcap, fps, section, save_dir, vis_dir, img_dir, save_test_dir, full_mask)

	pad_imgs(save_dir, img_dir, save_test_dir)





