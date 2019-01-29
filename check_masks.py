from pathlib import Path
import numpy as np
import cv2
import os.path


masks_dir = '012-APR-20-1-90_v3/masks'

for glob in Path(masks_dir).glob('*.png'):
	img_name = glob.parts[-1]
	img = cv2.imread(os.path.join(masks_dir, img_name))
	# print img.shape
	# print img.dtype
	# print
	if np.array_equal(img[...,0], img[...,1]) and np.array_equal(img[...,0], img[...,2]):
		print "bien"
	else:
		print "mal"
	diff_sets = set(np.unique(img)) - set([0,1,2])
	if diff_sets:
		print np.unique(img)


