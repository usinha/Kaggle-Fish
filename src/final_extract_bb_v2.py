from keras import backend as K
import cv2
import argparse
import numpy as np
from PIL import Image
import time
import imutils
import os
import glob
import random
from keras.models import load_model
# import the necessary packages
from keras.preprocessing import image as image_utils
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import  SGD
from keras.applications.inception_v3 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as VGG_preprocess_input
from keras.applications.vgg16  import decode_predictions as VGG_decode_predictions

#from imagenet_utils import decode_predictions
#from imagenet_utils import preprocess_input
#from vgg16 import VGG16
#FISH = ["SHARK", "HAMMERHEAD", "FISH", "RAY", "SALMON", "STURGEON", "DOLPHIN",
#     "great_white_shark", "platypus", "stingray", "'COHO", "ELECTRIC_RAY"]
#THRESHOLD_MATCHES = 4
THRESHOLD_PROB = 0.5
THRESHOLD_VGG_PROB = 0.25
TOP_SIZE = 3 # for each pyramid,size of top images based on probality it's a fish
FINAL_TOP_SIZE = 5 # for all pyramids
FISH = ["SHARK", "HAMMERHEAD", "FISH", "RAY", "SALMON", "STURGEON", "DOLPHIN",
     "GREAT_WHITE_SHARK", "PLATYPUS", "STINGRAY", "'COHO", "ELECTRIC_RAY",
     "TENCH", "BARRACOUTA","PUFFER", "GAR"]
THRESHOLD_MATCHES = 4
#MAX_OUT_IMAGES = 300
ROOT_WEIGHTS_DIR = '/home/icarus/kaggle/Kaggle-Fish/model_weights' #'/home/pyimagesearch/kaggle/source'
WEIGHTS_FILE = 'fish_bb_weights_2.h5'
#WEIGHTS_FILE_3 = 'fish_water_weights_2.h5'
total_out_images = 0
total_processed_images = 0
OUT_DIRECTORY = '/home/icarus/kaggle/Kaggle-Fish/data/Final_Derived_Images_bb/all_images' #"/home/pyimagesearch/kaggle/FINAL_DerivedImages"

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        ##
        for x in xrange(0, image.shape[1], stepSize):
            x_end = x + windowSize[0]
            if (x + windowSize[0] >= image.shape[1]) :
                x_end = image.shape[1]
            y_end =  y + windowSize[1]
            if (y + windowSize[1] >= image.shape[0]) :
                y_end = image.shape[0]
            # yield the current window
            if (y + windowSize[1] < image.shape[0]) or (x + windowSize[0] < image.shape[1]) :
                #yield the current window
                yield (x, y, image[y:y_end, x:x_end])
            #if (x + windowSize[0] >= image.shape[1]) :
            #    break;
        if (y + windowSize[1] >= image.shape[0]) :
            break;
#
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

def check_fish(obname) :

    match = False
    for f in FISH :
        if (obname.upper().find(f) > -1)    :
            match = True
            break
    return match
def VGG_predictIfFish(image_in, model, threshold_prob = 0.3) :
        # resize the image
        dim = (224,224)
        resized = cv2.resize(image_in, dim, interpolation=cv2.INTER_LINEAR)
        cv2_im = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        image = image_utils.img_to_array(pil_im)
        # our image is now represented by a NumPy array of shape (3, 224, 224),
        # but we need to expand the dimensions to be (1, 3, 224, 224) so we can
        # pass it through the network -- we'll also preprocess the image by
        # subtracting the mean RGB pixel intensity from the ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = VGG_preprocess_input(image)

        # classify the image
        #print("[INFO] classifying image...")
        preds = model.predict(image)
        #decodedTups = VGG_decode_predictions(preds, top=15)[0]
        decodedTups = VGG_decode_predictions(preds, top=15)[0]

        found = False
        pr = 0.0
        num_matches = 0
        for tup in decodedTups:
            if ((tup[1].upper() in FISH ) or check_fish(tup[1])) :
                #found = True
                pr += tup[2]
                num_matches += 1
                #print ("found fish:" + tup[1] + ":pr=" + str(tup[2]))
        #if (pr > threshold_prob):
        if (num_matches > THRESHOLD_MATCHES) or (pr > THRESHOLD_VGG_PROB):
            #print ("fish found:"  + " tot pr=" + str(pr))
            found = True
        return found,pr, num_matches
#
#  phase-2 (inception) prediction
#
def predictIfFish(image, model, threshold_prob = 0.5):  # resize the image
        dim = (299,299)
        threshold_prob = THRESHOLD_PROB
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
        cv2_im = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        im = image_utils.img_to_array(pil_im)
        # our image is now represented by a NumPy array of shape (3, 224, 224),
        # but we need to expand the dimensions to be (1, 3, 224, 224) so we can
        # pass it through the network -- we'll also preprocess the image by
        # subtracting the mean RGB pixel intensity from the ImageNet dataset
        im = np.expand_dims(im, axis=0)
        im = preprocess_input(im)
        found = False
        # classify the image
        #print("[INFO] classifying image...")
        pred = model.predict(im)
        pr = pred[0][0]
        if pr > threshold_prob :
            found = True
        return found, pr
#
#  phase-2 (inception) prediction
#
def predictIfFish_3(image, model, threshold_prob = 0.5) :
        # resize the image
        dim = (299,299)
        #  threshold_prob = THRESHOLD_PROB
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
        cv2_im = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        im = image_utils.img_to_array(pil_im)
        # our image is now represented by a NumPy array of shape (3, 224, 224),
        # but we need to expand the dimensions to be (1, 3, 224, 224) so we can
        # pass it through the network -- we'll also preprocess the image by
        # subtracting the mean RGB pixel intensity from the ImageNet dataset
        im = np.expand_dims(im, axis=0)
        im = preprocess_input(im)
        found = False
        # classify the image
        #print("[INFO] classifying image...")
        pred = model.predict(im)
        pr = pred[0][0]
        if pr >= threshold_prob :
            found = True
        return found, pr
#
# for an image, it extracts the top fish images
#
def extractFishImages(VGG_model,model, model_3,imagedir,filenm) :
    fullImagePath = os.path.join(imagedir,filenm)
    image = cv2.imread(fullImagePath)
    imagename = fullImagePath
    #OUT_DIRECTORY = "DerivedImages"
    if (not os.path.exists(OUT_DIRECTORY)) :
        os.mkdir(OUT_DIRECTORY)
    (winW, winH) = (150, 150)
    tot_wins = 0
    win_num = 0
    py_num = 0
    topItems = []
    topItem = None
    for resized in pyramid(image, scale=1.5,minSize=(180, 180)):
        indx = py_num
        t = []
        py_num += 1

        #print ('pyramid=' + str(py_num))
        #print(resized.shape)
            #if the image is too small, break from the loop
        if resized.shape[0] < winH or resized.shape[1] < winW:
            break
            # loop over the sliding window for each layer of the pyramid
        #cv2.imwrite("tempimages/copyImage" + str(i)+ ".jpg", resized)
        win_num = 0
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] < (winH - 16) or window.shape[1] <  (winW - 16) :
                continue

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            win_num += 1
            tot_wins +=1
            #if (tot_wins % 300 == 0 ) :
            #    print("total_win: {}, Pyramid#: {}".format(tot_wins, py_num))
            VGG_fishFound,VGG_pr, num_matches =  VGG_predictIfFish(window, VGG_model)
            if (py_num ==1 ) and (win_num == 1) :
		#initialize topItem
		topItem = (window, VGG_pr)
            if VGG_fishFound :
                fishFound, pr = predictIfFish(window, model)
		if fishFound:
		    if pr > topItem[1]:
		        topItem = (window, pr)
    ##
    global total_processed_images
    total_processed_images += 1
    filenm_only, ext = os.path.splitext(filenm)
    out_filenm = filenm_only + '_' + "p" + "{0:.2f}".format(topItem[1]) + ext
    full_out_file_name = os.path.join(OUT_DIRECTORY, out_filenm)
    print('processed ' + str(total_processed_images) + 'images -' + filenm)
    cv2.imwrite(full_out_file_name, topItem[0])


if __name__ == "__main__" :
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagedir", required = True,
        help = "path to imagedir")
    args = vars(ap.parse_args())
    imagedir = args["imagedir"]
    print ("dir=" + imagedir)
    # load phase-1 VGG model
    print("[INFO] loading VGG network...")
    VGG_model = VGG16(weights="imagenet")
    print("[INFO] model loaded...")
    #
    # load customized inception for phase-2

    #root_path = '/home/pyimagesearch/kaggle/source'
    #	

    weights_path = os.path.join(ROOT_WEIGHTS_DIR, WEIGHTS_FILE)
    #
    print("[INFO] loading Inception network...")
#    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
#                        input_tensor=None, input_shape=(299,299,3))
    # Note that the preprocessing of InceptionV3 is:
    # (x / 255 - 0.5) x 2
 #   print('Adding Average Pooling Layer and Softmax Output Layer ...')
#    output = InceptionV3_notop.output  # Shape: (8, 8, 2048)
#    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
 #   output = Flatten(name='flatten')(output)
 #   output = Dense(2, activation='softmax', name='predictions')(output)
 #   model_2 = Model(InceptionV3_notop.input, output)
    model_2 = load_model(weights_path)

   #
#    print ('loading saved weights')
#    model_2.load_weights(weights_path)
#    optimizer = SGD(lr = 0.0001, momentum = 0.9, decay = 0.0, nesterov = True)
#    print('compiling model2')
#    model_2.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    #
    #weights_path_3 = os.path.join(ROOT_WEIGHTS_DIR, WEIGHTS_FILE_3)
    print("[INFO] loading Inception network...")
    #model_3 = load_model(weights_path_3)
    model_3 = None
    print ('loading saved weights')
    #print("[INFO] model_3 loaded...")
    im_files = os.listdir(imagedir)
    im_files = sorted(im_files)
    #for file in os.listdir(imagedir):
    running_file_count = 0
    for file in im_files:
        running_file_count += 1
        if running_file_count < 521 :
	    continue
	if (running_file_count % 50) == 0 :          
            print ('running input file count=' + str(running_file_count))
        if file.endswith(".jpg") :
            extractFishImages(VGG_model,model_2,model_3,imagedir, file)


# classify the image
##print("[INFO] classifying image...")
##preds = model.predict(image)
##print('Predicted;', decode_predictions(preds, top=3)[0])
#(inID, label,probability) = decode_predictions(preds)[0][0]

# display the predictions to our screen
#print("ImageNet ID: {}, Label: {}".format(inID, label))
#cv2.putText(orig, "Label: {}".format(label), (10, 30),
#    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# cv2.imshow("Classification", orig)
#cv2.waitKey(0)
