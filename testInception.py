from keras import backend as K
import cv2
import argparse
import numpy as np
from PIL import Image
from skimage.transform import pyramid_gaussian
import time
import imutils
import os

# import the necessary packages
from keras.preprocessing import image as image_utils
##from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
#from imagenet_utils import decode_predictions
#from imagenet_utils import preprocess_input
#from vgg16 import VGG16
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
            #if (y + windowSize[1] < image.shape[0]) or (x + windowSize[0] < image.shape[1]) :
            # yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
#
def pyramid(image, scale=1.5, minSize=(30, 30)):
		# yield the original image
		yield image

		# keep looping over the pyramid
		while True:
			# compute the new dimensions of the image and resize it
			w = int(image.shape[1] / scale)
			image = imutils.resize(image, width=w)

			# if the resized image does not meet the supplied minimum
			# size, then stop constructing the pyramid
			if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
				break

			# yield the next image in the pyramid
			yield image
def predictIfFish(image, model) :
		# resize the image
		dim = (299,299)
		resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
		cv2_im = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
		cv2.imwrite("imagepredicted.jpg", cv2_im)
		pil_im = Image.fromarray(cv2_im)
		image = image_utils.img_to_array(pil_im)
		# our image is now represented by a NumPy array of shape (3, 224, 224),
		# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
		# pass it through the network -- we'll also preprocess the image by
		# subtracting the mean RGB pixel intensity from the ImageNet dataset
		image = np.expand_dims(image, axis=0)
		image = preprocess_input(image)

		# classify the image
		#print("[INFO] classifying image...")
		preds = model.predict(image)
		decodedTups = decode_predictions(preds, top=10)[0]

		print decodedTups
		#FISH = ["SHARK", "HAMMERHEAD", "FISH", "SALMON"]
		FISH = ["SHARK", "HAMMERHEAD", "FISH", "RAY", "SALMON", "STURGEON", "DOLPHIN"]
		found = False
		pr = 0.0
		for tup in decodedTups:
			for f in FISH :
				if (f.upper() in tup[1].upper()) :
					#found = True
					pr += tup[2]
					#print ("found fish:" + tup[1] + ":pr=" + str(tup[2]))
					#break
			#if (found):
				#break
		if (pr > 0.25):
			print ("fish found:"  + " tot pr=" + str(pr))
			found = True
		return found, pr
# main
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "path to image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
imagename = args["image"]
OUT_DIRECTORY = "tempimages"
if (not os.path.exists(OUT_DIRECTORY)) :
	os.mkdir(OUT_DIRECTORY)
filenmInd1 = imagename.find("img_")
folder = imagename[imagename.find("train")+5:filenmInd1]
fullfolder = OUT_DIRECTORY + folder[:-1]
print ("fullfolder=" + fullfolder)
if (not os.path.exists(fullfolder)) :
	os.mkdir(fullfolder)
filenmInd2 = imagename.find(".jpg")
filenm = imagename[filenmInd1:filenmInd2]
outFilePrefix = folder + filenm
print ("outfileprefix= " + outFilePrefix)


print("width:{} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
cv2.imshow("Image", image)

#cv2.waitKey(0)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#image = cv2.GaussianBlur(image,(5,5),0)
# image pyramid
# METHOD #2: Resizing + Gaussian smoothing.
(winW, winH) = (223, 223)
tot_wins = 0
win_num = 0
py_num = 0
print("[INFO] loading network...")
model = InceptionV3(weights="imagenet")
print("[INFO] model loaded...")
#for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
for resized in pyramid(image, scale=1.5,minSize=(128, 128)):
	py_num += 1
	# if the image is too small, break from the loop
	#if resized.shape[0] < winH or resized.shape[1] < winW:
#		break
    	# loop over the sliding window for each layer of the pyramid
	#cv2.imwrite("tempimages/copyImage" + str(i)+ ".jpg", resized)
	win_num = 0
	for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
		win_num += 1
		tot_wins +=1
		if (tot_wins % 100 == 0 ) :
			print("total_win: {}, Pyramid#: {}".format(tot_wins, py_num))
		fishFound,pr =  predictIfFish(window, model)
		if fishFound :
			fullOutFileName = OUT_DIRECTORY + outFilePrefix + "p" + "{0:.2f}".format(pr) \
				+ "-" +str(py_num) + "-" +  str(win_num)+  ".jpg"
			#fullOutFileName = "SHARKIM" + "-" + str(win_num)+ ".jpg"
			print (fullOutFileName)
			cv2.imwrite(fullOutFileName, window)

		# since we do not have a classifier, we'll just draw the window
		#clone = resized.copy()
		#cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		#cv2.imshow("Window", clone)
		##num = num + 1
		##cropped = clone[y: y + winH ,x:x + winW ]
		##if (num % 50 == 0):
		##	croppedResized = cv2.resize(cropped,(224, 224),cv2.INTER_CUBIC)
		##	cv2.imwrite("tempimages/newimage" + str(num) + ".jpeg",croppedResized)
		##	#cv2.imwrite("tempimages/resizedimage" + str(num) + ".jpeg",resized)
		##	cv2.imshow("Window", croppedResized)
		##	cv2.waitKey(1)
		#time.sleep(0.025)
		#
		##print("[INFO] loading and preprocessing image...")
	# show the resized image
	#cv2.imshow("Layer {}".format(i + 1), resized)
    #cv2.waitKey(0)
# load the input image using the Keras helper utility while ensuring
# that the image is resized to 224x224 pxiels, the required input
# dimensions for the network -- then convert the PIL image to a
# NumPy array
##print("[INFO] loading and preprocessing image...")
##image = image_utils.load_img(args["image"], target_size=(224, 224))
##image = image_utils.load_img("tempimages/newimage850.jpeg", target_size=(224, 224))
##image = image_utils.img_to_array(image)
# our image is now represented by a NumPy array of shape (3, 224, 224),
# but we need to expand the dimensions to be (1, 3, 224, 224) so we can
# pass it through the network -- we'll also preprocess the image by
# subtracting the mean RGB pixel intensity from the ImageNet dataset
##image = np.expand_dims(image, axis=0)
##image = preprocess_input(image)
# load the VGG16 network
##print("[INFO] loading network...")
##model = VGG16(weights="imagenet")

# classify the image
##print("[INFO] classifying image...")
##preds = model.predict(image)
##print('Predicted;', decode_predictions(preds, top=3)[0])
#(inID, label,probability) = decode_predictions(preds)[0][0]

# display the predictions to our screen
#print("ImageNet ID: {}, Label: {}".format(inID, label))
#cv2.putText(orig, "Label: {}".format(label), (10, 30),
#	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# cv2.imshow("Classification", orig)
#cv2.waitKey(0)
