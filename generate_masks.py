import utils
from utils import homography, warp_image, string2msec, msec2string, merge_images, get_offset, undistort, pad_img
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

def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--json_path', type=str, required=True)
	parser.add_argument('--save_dir', type=str, required=True)
	parser.add_argument('--annotations_dir', type=str, required=True)
	parser.add_argument('--labels', type=str, required=True)
	parser.add_argument('--vis_dir', type=str, required=True)
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
	mask = PIL.Image.fromarray(mask)
	draw = PIL.ImageDraw.Draw(mask)

	with open(annotation_path, 'r') as f:
		json_data = json.load(f)

	for shape in json_data["shapes"]:

		npoints = len(shape["points"])
		xy = [check_point(tuple(point), sz) for point in shape["points"]]
		label_mapping = labels_mapping[shape["label"]]

		if npoints == 2:
			draw.line(xy=xy, fill=label_mapping, width=10)
		elif npoints > 2:
			draw.polygon(xy=xy, fill=label_mapping)

	return np.array(mask)

def select_section(sections, section_name):

	section_start = section_name.split("-")[0]
	section_end = section_name.split("-")[1]

	for section in sections:
		if (section['start'] == section_start) and (section['end'] == section_end):
			return section
	return None


def preprocess_img(img):

	img = undistort(img)
	img_bgr_down = downscale(img[...,:-1], 1000, False, pad_img=False)
	alpha_channel_down = downscale(img[...,-1], 1000, True, pad_img=False)
	new_img = np.dstack((img_bgr_down, alpha_channel_down))

	return new_img

def fix_mask(warped, mask):

	warped = warped[:mask.shape[0], :mask.shape[1]]
	new_mask = mask.copy()
	new_mask[warped[...,-1] != 255] = 255
	warped = warped[...,:-1]

	return warped, new_mask


def generate_masks_section(vidcap, fps, section, save_dir, vis_dir, full_mask):


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

	M = homography(frame_start[...,:-1], frame_end[...,:-1], draw_matches=False)
	warped_end, _ = warp_image(frame_end, M, alpha_channel=False)

	offset_start_x, offset_start_y = offset_start
	offset_end_x, offset_end_y = offset_end
	
	mask_start = full_mask[offset_start_y:offset_start_y + frame_start.shape[0], offset_start_x:offset_start_x + frame_start.shape[1]]
	frame_start, mask_start = fix_mask(frame_start, mask_start)

	vis_start = vis.vis_seg(frame_start, mask_start, vis.make_palette(3))
	cv2.imwrite(os.path.join(save_dir, "{}.png".format(section["start"])), pad_img(mask_start, 255))
	cv2.imwrite(os.path.join(vis_dir, "{}.png".format(section["start"])), vis_start)
	
	mask_end = full_mask[offset_end_y:offset_end_y + warped_end.shape[0], offset_end_x:offset_end_x + warped_end.shape[1]]
	warped_end, mask_end = fix_mask(warped_end, mask_end)

	vis_end = vis.vis_seg(warped_end, mask_end, vis.make_palette(3))
	cv2.imwrite(os.path.join(save_dir, "{}.png".format(section["end"])), pad_img(mask_end, 255))
	cv2.imwrite(os.path.join(vis_dir, "{}.png".format(section["end"])), vis_end)

	success = True
	current_time = start_time_msec + delay_msec
	while success and (current_time < end_time_msec):
	    vidcap.set(cv2.CAP_PROP_POS_MSEC,(current_time))
	    success, frame_next = vidcap.read()
	    if success:

	    	frame_next = undistort(frame_next)
	        frame_next = downscale(frame_next, 1000, False, pad_img=False)
	        M = homography(frame_start, frame_next, draw_matches=False)

	        warped, shift = warp_image(frame_next, M, alpha_channel=False)
	        h, w, z = warped.shape
	        offset_x, offset_y = shift
	        offset_x += offset_start_x
	        offset_y += offset_start_y
	        offset_x, offset_y = get_offset((offset_x, offset_y))
	        mask = full_mask[offset_y:offset_y + h, offset_x:offset_x + w]
	        warped, mask = fix_mask(warped, mask)

	        vis_img = vis.vis_seg(warped, mask, vis.make_palette(3))
	        cv2.imwrite(os.path.join(save_dir, "{}.png".format(msec2string(current_time))), pad_img(mask, 255))
	        cv2.imwrite(os.path.join(vis_dir, "{}.png".format(msec2string(current_time))), vis_img)
	        
	    current_time += delay_msec



if __name__ == "__main__":

	args = make_parser().parse_args()

	with open(args.json_path, 'r') as f:
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

	for glob in Path(args.annotations_dir).glob('*.json'):

		section_name = os.path.splitext(os.path.basename(glob.parts[-1]))[0]

		img_section = cv2.imread(os.path.join(merged_dir, section_name + '.png'))
		full_mask = json2mask(os.path.join(args.annotations_dir, glob.parts[-1]), labels_mapping, sz=img_section.shape[:2])
		
		section = select_section(sections, section_name)
		generate_masks_section(vidcap, fps, section, args.save_dir, args.vis_dir, full_mask)





