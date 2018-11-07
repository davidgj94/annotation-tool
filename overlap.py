from utils import homography, warp_image, string2msec, msec2string, merge_images
import os.path
import os
import json
import cv2
from Tkinter import * 
from matplotlib import pyplot as plt
import argparse
import numpy as np
from downscale import _downscale as downscale


def color_img(img, color):
	img = np.float32(img)
	img *= 1. - 0.5
	img += 0.5 * np.array(color)
	return np.uint8(img)


class HelperClass:

	def __init__(self, video_path, start_time, fps, json_path):
		self.index = 0
		self.M_list = []
		self.start_time_msec = string2msec(start_time)
		self.delay_msec = int(1000 * (1 / fps))
		self.vidcap = cv2.VideoCapture(video_path)
		self.vidcap.set(cv2.CAP_PROP_POS_MSEC,(self.start_time_msec))
		success, self.frame_start = self.vidcap.read()
		self.frame_start = downscale(self.frame_start, 1000, False, pad_img=False)
		self.frame_start_red = color_img(self.frame_start, [0, 0, 255])
		self.json_path = json_path
		if not os.path.exists(json_path):
			os.mknod(json_path)
			json_data = dict()
			json_data['fps'] = fps
			json_data['video_path'] = video_path
			json_data['sections'] = []
			with open(json_path, 'w') as f:
				json.dump(json_data, f)

	def plot_overlap(self):

		current_time = self.start_time_msec + self.index * self.delay_msec
		self.vidcap.set(cv2.CAP_PROP_POS_MSEC,(current_time))
		success, frame_next = self.vidcap.read()
		if success:
			frame_next = downscale(frame_next, 1000, False, pad_img=False)
			if len(self.M_list) < self.index:
				M = homography(self.frame_start, frame_next, draw_matches=False)
				self.M_list.append(M)
			else:
				M = self.M_list[self.index - 1]

		 	warped, shift = warp_image(frame_next, M, alpha_channel=False)	
	        warped_blue = color_img(warped, [255, 0, 0])

	        overlapped_img, _, _ = merge_images(self.frame_start_red, warped_blue, shift)

	        plt.figure()
	        plt.imshow(overlapped_img[...,::-1])
	        plt.show()
	        
	        print msec2string(current_time)

	def forward(self):
		plt.close('all')
		self.index += 1
		self.plot_overlap()

	def backward(self):
		if self.index > 1:
			plt.close('all')
			self.index -= 1
			self.plot_overlap()

	def exit(self,root):
		plt.close('all')
		if self.index > 0:
			self.save()
		root.destroy()

	def save(self):

		section = dict()
		section['start'] = msec2string(self.start_time_msec)
		section['end'] = msec2string(self.start_time_msec + self.index * self.delay_msec)

		with open(self.json_path, 'r+') as f:
			json_data = json.load(f)
			json_data['sections'].append(section)
			f.seek(0)
			json.dump(json_data, f)

	def save_and_continue(self):

		if self.index > 0:
			self.save()
			self.start_time_msec = self.start_time_msec + self.index * self.delay_msec
			self.index = 0
			self.M_list = []
			self.vidcap.set(cv2.CAP_PROP_POS_MSEC,(self.start_time_msec))
			success, self.frame_start = self.vidcap.read()
			self.frame_start = downscale(self.frame_start, 1000, False, pad_img=False)
			self.frame_start_red = color_img(self.frame_start, [0, 0, 255])



def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_path', type=str, required=True)
	parser.add_argument('--start_time', type=str, required=True)
	parser.add_argument('--json_path', type=str, required=True)
	parser.add_argument('--fps', type=float, default=1.0)
	return parser

if __name__ == "__main__":

	args = make_parser().parse_args()
	helper = HelperClass(args.video_path, args.start_time, args.fps, args.json_path)
	root = Tk()
	button1 = Button(root, text='Atras', command=lambda:helper.backward())
	button1.pack(side=LEFT)
	button2 = Button(root, text='Adelante', command=lambda:helper.forward())
	button2.pack(side=LEFT)
	button3 = Button(root, text='Terminar', command=lambda:helper.exit(root))
	button3.pack(side=LEFT)
	button4 = Button(root, text='Continuar', command=lambda:helper.save_and_continue())
	button4.pack(side=LEFT)
	root.mainloop()









