import cv2
from os.path import join
import os
from imgaug import augmenters as iaa

augmented_image_dir = "bird_dataset_crop_2/train_images/"

species = [
    '004.Groove_billed_Ani',
'009.Brewer_Blackbird',
'010.Red_winged_Blackbird',
'011.Rusty_Blackbird',
'012.Yellow_headed_Blackbird',

'013.Bobolink',
'014.Indigo_Bunting',
'015.Lazuli_Bunting',
'016.Painted_Bunting',

'019.Gray_Catbird',
'020.Yellow_breasted_Chat',
'021.Eastern_Towhee',
'023.Brandt_Cormorant',
'026.Bronzed_Cowbird',
'028.Brown_Creeper',
'029.American_Crow',
'030.Fish_Crow',
'031.Black_billed_Cuckoo',
'033.Yellow_billed_Cuckoo',

'034.Gray_crowned_Rosy_Finch']
#### Image Augmentation
#
# Author: Evan Juras
# Date: 5/15/20
# Description:
# This program takes a set of original images and creates augmented images that can be used
# as additional training data for an image classification model. It loads each original image,
# creates N randomly augmented images, and saves the new images.

import imgaug as ia
from imgaug import augmenters as iaa
import os
import sys
import argparse
from glob import glob
import cv2


## Define control variables and parse user inputs
parser = argparse.ArgumentParser()
parser.add_argument('--imgdir', help='Folder containing images to augment',default='bird_dataset_cropc/train_images/')
                    
parser.add_argument('--imgext', help='File extension of images (for example, .JPG)',
                    default='.jpg')
parser.add_argument('--numaugs', help='Number of augmented images to create from each original image',
                    default=2)
parser.add_argument('--debug', help='Displays every augmented image when enabled',
                    default=False)

args = parser.parse_args()

#### Define augmentation sequence ####
# This can be tweaked to create a huge variety of image augmentations.
# See https://github.com/aleju/imgaug for a list of augmentation techniques available.
seq1 = iaa.Sequential([
                                 # Horizontal flip 50% of images
    iaa.Fliplr(0.5),               # Crop all images between 0% to 10%
    iaa.GaussianBlur(sigma=(0, 1)),              # Add slight blur to images
    iaa.Multiply((0.7, 1.3), per_channel=0.2),   # Slightly brighten, darken, or recolor images
##    iaa.Affine(
##        scale={"x": (0.8, 1.2), "y": (0.8,1.2)},                # Resize image
##        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # Translate image
##        rotate=(-5, 5),                                         # Rotate image
##        mode=ia.ALL, cval=(0,255)                               # Filling in extra pixels
##        )
    ])
hue =iaa.Sequential([
        iaa.Sometimes(0.3, iaa.Multiply((0.5, 1.5), per_channel=0.5)),
        
        iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
        iaa.Sometimes(0.1, iaa.MotionBlur(k=15, angle=[-45, 45])),
        iaa.Sometimes(0.4, iaa.MultiplyHue((0.5, 1.5))),
        iaa.Sometimes(0.3, iaa.MultiplySaturation((0.5, 1.5))),
        iaa.Sometimes(0.1, iaa.MultiplyHueAndSaturation((0.5, 1.5),
                                                         per_channel=True)),
        iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0))),
        iaa.Sometimes(0.3, iaa.ChangeColorTemperature((1100, 10000))),
        iaa.Sometimes(0.1, iaa.GammaContrast((0.5, 2.0))),
        iaa.Sometimes(0.2, iaa.SigmoidContrast(gain=(3, 10),
                                               cutoff=(0.4, 0.6)))
        
        
        ,
        

                
                ])
#### Start of main program ####

# Obtain list of images in IMG_DIR directory

j=0
def augm(img_fns):
# Go through every image in directory, augment it, and save new image/annotation data
    for img_fn in img_fns:

        #---- Load image ----#
        img1_bgr = cv2.imread(img_fn) # Load image with OpenCV
        img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB) # Re-color to RGB from BGR

        #---- Augment image N times----#
        for i in range(NUM_AUG_IMAGES):
            img_aug1 = seq1(images=[img1])[0] # Apply augmentation to image
            img_aug1 = hue(images=[img_aug1])[0]
            #---- Save image ----#
            base_fn = img_fn.replace(IMG_EXTENSION,'') # Get image base filename
            img_aug_fn = base_fn + ('_aug%d' % (i+1)) + IMG_EXTENSION # Append "aug#" to filename
            img_aug_bgr1 = cv2.cvtColor(img_aug1, cv2.COLOR_RGB2BGR) # Re-color to BGR from RGB
            cv2.imwrite(img_aug_fn,img_aug_bgr1) # Save image to disk

            # Display original and augmented images (if debug is enabled)
            if debug:
                cv2.imshow('Original image',img1_bgr)
                cv2.imshow('Augmented image',img_aug_bgr1)
                cv2.waitKey()

    # If images were displayed, close them at the end of the program
    if debug:
        cv2.destroyAllWindows()

for x in species:
    IMG_DIR = args.imgdir + x
    
    if j == 1 or j == 3 or j == 14 or j == 15 or j == 16 or j == 18 :
        if not os.path.isdir(IMG_DIR):
            print('%s is not a valid directory.' % IMG_DIR)
            sys.exit(1)
        IMG_EXTENSION = args.imgext
        NUM_AUG_IMAGES = int(args.numaugs)
        debug = bool(args.debug)
        cwd = os.getcwd()
        img_fns = glob(IMG_DIR + '/*' + IMG_EXTENSION)
        augm(img_fns)
    j=j+1
augm('bird_dataset_cropc/train_images/')