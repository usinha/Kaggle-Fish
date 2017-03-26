from keras import backend as K
from keras.models import load_model
import cv2
import argparse
import numpy as np
from PIL import Image
import time
import imutils
import os
import glob
import random

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

THRESHOLD_PROB = 0.3
THRESHOLD_VGG_PROB = 0.2
TOP_SIZE = 2 # size of top images based on probality it's a fish
FISH = ["SHARK", "HAMMERHEAD", "FISH", "RAY", "SALMON", "STURGEON", "DOLPHIN",
     "GREAT_WHITE_SHARK", "PLATYPUS", "STINGRAY", "'COHO", "ELECTRIC_RAY",
     "TENCH", "BARRACOUTA","PUFFER", "GAR"]
THRESHOLD_MATCHES = 4
MAX_OUT_IMAGES = 200
ROOT_MODEL_DIR = '/home/icarus/kaggle/Kaggle-Fish/model_weights'
MODEL_FILE = 'fish_weights_2.h5'
MODEL_FILE_3 = 'fish_water_weights_2.h5'
OUT_DIRECTORY = '/home/icarus/kaggle/Kaggle-Fish/data/DerivedImages'

total_out_images = 0

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
def predictIfFish(image, model, threshold_prob = 0.4) :
        # resize the image
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
def extractFishImages(VGG_model,model, model_3,fullImagePath) :
    image = cv2.imread(fullImagePath)
    imagename = fullImagePath
    if (not os.path.exists(OUT_DIRECTORY)) :
        os.mkdir(OUT_DIRECTORY)
    filenmInd1 = imagename.find("img_")
    folder = imagename[imagename.find("train")+5:filenmInd1]
    fullfolder = OUT_DIRECTORY + folder[:-1]
    #print ("fullfolder=" + fullfolder)
    if (not os.path.exists(fullfolder)) :
        os.mkdir(fullfolder)
    filenmInd2 = imagename.find(".jpg")
    filenm = imagename[filenmInd1:filenmInd2]
    outFilePrefix = folder + filenm
    #print ("outfileprefix= " + outFilePrefix)
    #print("width:{} pixels".format(image.shape[1]))
    #print("height: {} pixels".format(image.shape[0]))
    #cv2.imshow("Image", image)

    #cv2.waitKey(0)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = cv2.GaussianBlur(image,(5,5),0)
    # image pyramid
    # METHOD #2: Resizing + Gaussian smoothing.
    topItems = []
    (winW, winH) = (150, 150)
    tot_wins = 0
    win_num = 0
    py_num = 0
    #print("[INFO] loading network...")
    #model = VGG16(weights="imagenet")
    #print("[INFO] model loaded...")
    #for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
    topItems = []
    for resized in pyramid(image, scale=1.5,minSize=(180, 180)):
        indx = py_num
        t = []
        topItems.append(t)
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
            if (tot_wins % 300 == 0 ) :
                print("total_win: {}, Pyramid#: {}".format(tot_wins, py_num))
            VGG_fishFound,VGG_pr, num_matches =  VGG_predictIfFish(window, VGG_model)
            if VGG_fishFound :
                fishFound, pr = predictIfFish(window, model)
                if fishFound :
                    fishFound, pr_3 = predictIfFish_3(window, model_3)
                    if fishFound:
                         fullOutFileName = OUT_DIRECTORY + outFilePrefix + "-" + str(num_matches) \
                             + "-" + str(py_num) \
                             + "p" + "{0:.2f}".format(VGG_pr) \
                             + "p" + "{0:.2f}".format(pr) \
                             + "p" + "{0:.2f}".format(pr_3) \
                             + ".jpg"

                         tup = (pr + VGG_pr, window, fullOutFileName)
                         topItems[indx].append(tup)
                         #print ("appended image")
                         if len(topItems[indx]) > TOP_SIZE :
                             topItems[indx] = sorted(topItems[indx],key = lambda t:t[0], reverse= True)[0:TOP_SIZE]
    ##
    global total_out_images
    for t in topItems:
        for fish in t:
            total_out_images += 1
            if total_out_images < MAX_OUT_IMAGES :
                #print ("output .. " + fish[2] )
                cv2.imwrite(fish[2], fish[1])
                if (total_out_images % 25) == 0:
	     	    print("output.." + fish[2])


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
    model_path = os.path.join(ROOT_MODEL_DIR, MODEL_FILE)
    print("[INFO] loading Inception network...")
    model_2 = load_model(model_path)
    #
    model_path_3 = os.path.join(ROOT_MODEL_DIR, MODEL_FILE_3)
    print("[INFO] loading Inception network...")
    model_3 = load_model(model_path_3)
    print("[INFO] model_3 loaded...")
    im_files = os.listdir(imagedir)
    random.seed()
    random.shuffle(im_files)
    #for file in os.listdir(imagedir):
    for file in im_files:
        if total_out_images > MAX_OUT_IMAGES :
            break
        if file.endswith(".jpg") :
            extractFishImages(VGG_model,model_2,model_3,imagedir + "/" + file)


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
