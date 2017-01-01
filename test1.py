from __future__ import print_function
import argparse
import cv2
import numpy as np
from keras.preprocessing import image as image_utils
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "path to image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
print("width:{} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
#cv2.imshow("Image", image)
cv2_im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#pil_im = Image.open(args["image"])
pil_im = Image.fromarray(cv2_im)
pil_im.show()
cv2.waitKey(0)
cv2.imwrite("copyImage.jpg", image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image,(5,5),0)
#cv2.imshow("Blurred",image)
canny = cv2.Canny(image,30,150)
#cv2.imshow("Canny", canny)
#cv2.waitKey(0)
(_,cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
print("number of countours= {}".format(len(cnts)))
#
# bounding box
rectWithArea = []
for cnt in cnts :
	rect = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	area = box[1][0]* box[1][1]
	rectWithArea.append((box,area))
sortedTups = sorted(rectWithArea, key=lambda t :t[1], reverse=True)
sortedBoxes = [t[0] for t in sortedTups]

#cntWithArea = []
#for cnt in cnts :
#	area = cv2.contourArea(cnt)
#	cntWithArea.append((cnt,area))
#sortedTups = sorted(cntWithArea, key=lambda t :t[1], reverse=True)
#sortedCnts = [t[0] for t in sortedTups]
#for i in range(10):
#	print (sortedTups[i][1])
#
fish = image.copy()

cv2.drawContours(fish, sortedBoxes, -1, (0,255,0), 2)

#cv2.imshow("fish", fish)
cv2.waitKey(0)
