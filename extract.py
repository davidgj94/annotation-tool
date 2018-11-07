import argparse
import json
import pdb
from utils import string2msec, msec2string
import cv2
import os.path
import os

def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--json_path', type=str, required=True)
	parser.add_argument('--save_dir', type=str, required=True)
	return parser

def extract_time(sections):
	start_times = [string2msec(s["start"]) for s in sections]
	end_times = [string2msec(s["end"]) for s in sections]
	return list(set(start_times + end_times))

if __name__ == "__main__":

	args = make_parser().parse_args()
	with open(args.json_path, 'r') as f:
		json_data = json.load(f)

	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)

	fps = float(json_data['fps'])
	video_path = json_data['video_path']
	sections = json_data['sections']
	frames_time = extract_time(sections)

	vidcap = cv2.VideoCapture(video_path)

	for ftime in frames_time:
		vidcap.set(cv2.CAP_PROP_POS_MSEC,(ftime))
		success, frame = vidcap.read()
		if success:
			cv2.imwrite(os.path.join(args.save_dir, "{}.png".format(msec2string(ftime))), frame)
	
