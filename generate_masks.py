from utils import homography, warp_image, string2msec, msec2string, merge_images, get_offset
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
    

def json2mask(annotation_path, save_dir, labels_mapping, sz=(2160, 4096)):

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
			draw.line(xy=xy, fill=label_mapping, width=20)
		elif npoints > 2:
			draw.polygon(xy=xy, fill=label_mapping)

	mask_name = os.path.splitext(os.path.basename(annotation_path))[0] + ".png"
	mask.save(os.path.join(save_dir, mask_name))
	

def generate_masks_section(vidcap, fps, section, save_dir):

	start_time_msec = string2msec(section["start"])
	end_time_msec = string2msec(section["end"])
	delay_msec = int(1000 * (1 / fps))

	vidcap.set(cv2.CAP_PROP_POS_MSEC,(start_time_msec))
	success, frame_start = vidcap.read()
	frame_start = downscale(frame_start, 1000, False, pad_img=False)
	mask_start = cv2.imread(os.path.join(save_dir, "{}.png".format(section["start"])))
	mask_start = downscale(mask_start, 1000, True, pad_img=False)

	vidcap.set(cv2.CAP_PROP_POS_MSEC,(end_time_msec))
	success, frame_end = vidcap.read()
	frame_end = downscale(frame_end, 1000, False, pad_img=False)
	mask_end = cv2.imread(os.path.join(save_dir, "{}.png".format(section["end"])))
	mask_end = downscale(mask_end, 1000, True, pad_img=False)

	M = homography(frame_start, frame_end, draw_matches=False)
	mask_end_warped, shift = warp_image(mask_end, M, alpha_channel=False, is_mask=False)
	full_mask, offset_start, offset_end = merge_images(mask_start, mask_end_warped, shift, blend=False)
	full_mask = np.uint8(full_mask)
	vis_img = vis.vis_seg(full_mask, full_mask[...,0], vis.make_palette(3))
	offset_start_x, offset_start_y = offset_start

	success = True
	current_time = start_time_msec + delay_msec
	while success and (current_time < end_time_msec):
	    vidcap.set(cv2.CAP_PROP_POS_MSEC,(current_time))
	    success, frame_next = vidcap.read()
	    if success:
	        frame_next = downscale(frame_next, 1000, False, pad_img=False)
	        M = homography(frame_start, frame_next, draw_matches=False)
	        warped, shift = warp_image(frame_next, M, alpha_channel=False)
	        h, w, z = warped.shape
	        offset_x, offset_y = shift
	        offset_x += offset_start_x
	        offset_y += offset_start_y
	        offset_x, offset_y = get_offset((offset_x, offset_y))
	        mask = full_mask[offset_y:offset_y + h, offset_x:offset_x + w]
	        vis_img = vis.vis_seg(warped[...,::-1], mask[...,0], vis.make_palette(3))
	        plt.figure()
	        plt.imshow(vis_img)
	        plt.show()
	        cv2.imwrite(os.path.join(save_dir, "{}.png".format(msec2string(current_time))), mask)
	    current_time += delay_msec



if __name__ == "__main__":

	args = make_parser().parse_args()

	with open(args.json_path, 'r') as f:
		json_data = json.load(f)

	fps = float(json_data['fps'])
	video_path = json_data['video_path']
	sections = json_data['sections']

	with open(args.labels, 'r') as f:
		labels_mapping = dict()
		for line in f:
			line = line.replace(" ", "")
			label_name = line.split(':')[0]
			label_integer = int(line.split(':')[1])
			labels_mapping[label_name] = label_integer

	for glob in Path(args.annotations_dir).glob('*.json'):
		json2mask(os.path.join(args.annotations_dir, glob.parts[-1]), args.save_dir, labels_mapping)

	vidcap = cv2.VideoCapture(video_path)

	for section in sections:
		generate_masks_section(vidcap, fps, section, args.save_dir)



