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
		self.frame_start_red = color_img(self.frame_start, [0, 0, 255])
		if not os.path.exists(json_path):
			os.mknod(json_path)
			self.json_data = dict()
			self.json_data['fps'] = fps
			self.json_data['video_path'] = video_path
			self.json_data['sections'] = []
			with open(json_path, 'w') as f:
				json.dump(self.json_data, f)
		else:
			with open(json_path) as f:
				self.json_data = json.load(f)

	def prueba1(self):
		plt.close('all')
		plt.figure()
		plt.imshow(self.frame_start[...,::-1])
		plt.show()

	def prueba2(self):
		plt.close('all')
		plt.figure()
		plt.imshow(self.frame_start_red[...,::-1])
		plt.show()

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
		self.index += 1
		plt.close('all')
		self.plot_overlap()

	def backward(self):
		if self.index > 1:
			self.index -= 1
			plt.close('all')
			self.plot_overlap()

	def exit(self,root):
		section = dict()
		section['start']
		self.json_data['sections'].append()
		root.destroy()



def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_path', type=str, required=True)
	parser.add_argument('--start_time', type=str, required=True)
	#parser.add_argument('--json_path', type=str, required=True)
	parser.add_argument('--fps', type=float, default=1.0)
	return parser

if __name__ == "__main__":

	args = make_parser().parse_args()
	helper = HelperClass(args.video_path, args.start_time, args.fps)
	root = Tk()
	button1 = Button(root, text='Atras', command=lambda:helper.backward())
	button1.pack(side=LEFT)
	button2 = Button(root, text='Adelante', command=lambda:helper.forward())
	button2.pack(side=LEFT)
	button3 = Button(root, text='Terminar', command=lambda:helper.exit())
	button3.pack(side=LEFT)
	root.mainloop()
    # button.bind('<ButtonPress-1>', lambda event, direction=direction: start_motor(direction))
    # button.bind('<ButtonRelease-1>', lambda event: stop_motor())
    

	# args = parser.parse_args()
	# start_time_msec = string2msec(args.start_time)
	# delay_msec = int(1000 * (1 / args.fps))

	# if not os.path.exists(args.json_path):
	# 	os.mknod(args.json_path)
	# 	json_data = dict()
	# 	json_data['fps'] = args.fps
	# 	json_data['video_path'] = args.video_path
	# 	with open(args.json_path, 'w') as f:
 #    		json.dump(json_data, f)
 #    else:
	# 	with open(args.json_path) as f:
	# 		json_data = json.load(f)

	# start_time_msec = string2msec(args.start_time)
	# delay_msec = int(1000 * (1 / args.fps))

	# vidcap = cv2.VideoCapture(args.video_path)

	# vidcap.set(cv2.CAP_PROP_POS_MSEC,(start_time_msec))
	# success, frame_start = vidcap.read()
	# if success:
	# 	frame_start = downscale(frame_start, 1000, False, pad_img=False)
	# 	frame_start_ = np.float32(frame_start)
	# 	frame_start_ *= 1. - 0.5
	# 	frame_start_ += 0.5 * np.array([0,0,255])
	# 	frame_start_ = np.uint8(frame_start_)

	# index = 0
	# M_list = []
	# current_time = start_time_msec + delay_msec
	# while True:

	# 	if forward:
	# 		index += 1
	# 	 elif backward:
	#  		index -= 1

 #    	current_time = start_time_msec + index * delay_msec
 #    	vidcap.set(cv2.CAP_PROP_POS_MSEC,(current_time))
	#     success, frame_next = vidcap.read()
	#     if success:
	#         frame_next = downscale(frame_next, 1000, False, pad_img=False)
	#         if length(M) < index:
	#  			M = homography(frame_start, frame_next, draw_matches=False)
	#  			M_list.append(M)
	#  		else:
	#  			M = M_list[index]
	# 	 	warped, shift = warp_image(frame_next, M, alpha_channel=False)	

	#         warped_ =  np.float32(warped)
	#         warped_ *= 1. - 0.5
	#         warped_ += 0.5 * np.array([255,0,0])
	#         warped_ = np.uint8(warped_)

	#         overlap_img, _, _ = merge_images(frame_start_, warped_, shift)
	#         plt.figure()
	#         plt.imshow(overlap_img[...,::-1])
	#         plt.show()
	        
	#         print msec2string(current_time)
	#         current_time += delay_msec









