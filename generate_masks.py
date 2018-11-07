from utils import homography, warp_image, string2msec, msec2string, merge_images
import json
import cv2
import os.path

def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--json_path', type=str, required=True)
	parser.add_argument('--save_dir', type=str, required=True)
	return parser

def generate_masks_section(vidcap, fps, section, save_dir):

	start_time_msec = string2msec(section["start"])
	end_time_msec = string2msec(section["end"])
	delay_msec = int(1000 * (1 / fps))

	vidcap.set(cv2.CAP_PROP_POS_MSEC,(start_time_msec))
	success, frame_start = vidcap.read()
	frame_start = downscale(frame_start, 1000, False, pad_img=False)
	mask_start = cv2.imread(os.path.join(save_dir, "{}.png".format(section["start"])))
	mask_start[...,0] = downscale(mask_start, 1000, True, pad_img=False)

	vidcap.set(cv2.CAP_PROP_POS_MSEC,(end_time_msec))
	success, frame_end = vidcap.read()
	frame_end = downscale(frame_end, 1000, False, pad_img=False)
	mask_end = cv2.imread(os.path.join(save_dir, "{}.png".format(section["end"])))
	mask_end[...,0] = downscale(mask_end, 1000, True, pad_img=False)

	M = homography(frame_start, frame_end, draw_matches=False)
	mask_end_warped, shift = warp_image(mask_end, M, alpha_channel=False, is_mask=False)
	full_mask, offset_start, offset_end = merge_images(mask_start, mask_end_warped, shift, blend=False)
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
	        mask = full_mask[offset_y:offset_y + h, offset_x:offset_x + w]
	        cv2.imwrite(os.path.join(save_dir, "{}.png".format(msec2string(current_time))), mask)
	    current_time += delay_msec



if __name__ == "__main__":

	args = make_parser().parse_args()

	with open(args.json_path, 'r') as f:
		json_data = json.load(f)

	fps = float(json_data['fps'])
	video_path = json_data['video_path']
	sections = json_data['sections']

	vidcap = cv2.VideoCapture(video_path)

	for section in sections:
		generate_masks_section(vidcap, fps, section, args.save_dir)



