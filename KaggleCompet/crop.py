from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN_ResNet50_FPN_Weights
import torchvision

import matplotlib.pyplot as plt
from PIL import Image,ImageFile

import os 




weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
model2=torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="COCO_V1")

# Put the model in inference mode
model.eval()


model2.eval()



def save_img(bboxes,full_path):
    """Saving images
    """
    img = Image.open(full_path)
    left, top , right, bottom  = bboxes
    cropped = img.crop( ( left, top, right, bottom) )  # size: 45, 45
    try:
        cropped.save(full_path, "JPEG", quality=80, optimize=True, progressive=True)
    except IOError:
        ImageFile.MAXBLOCK = cropped.size[0] * cropped.size[1]
        cropped.save(full_path, "JPEG", quality=80, optimize=True, progressive=True)
        
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
weights2= FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
def compare_model(model, img,weights):
    """"Compares two models for cropping images
    """
    preprocess = weights.transforms()
    batch = [preprocess(img)]
    prediction = model(batch)[0]
    bboxes0, labels0, scores0 = prediction["boxes"],[weights.meta["categories"][i] for i in prediction["labels"]],prediction["scores"]
    bboxes0,labels0,scores0= bboxes0[0].tolist(),labels0[0],scores0[0].tolist()
    print(scores0)
    if len(bboxes0)>0:
        if labels0=='bird':
            maximum_proba=scores0
            
            
            
            
            return maximum_proba, bboxes0
    return(-1,-1)

def main_crop(path):
    """produces cropped images of the bird dataset
    """
    for dossier, sous_dossiers, fichiers in os.walk(path):
        print (sous_dossiers)
        for num, fichier in enumerate(fichiers):
            if num%50==0:
                print(num)
            full_path = os.path.join(dossier, fichier)
            img = read_image(full_path)
            maximum0, bboxes_f1 = compare_model(model, img,weights)
            maximum1, bboxes_f2 = compare_model(model2, img,weights2)
            
            
            if max([maximum0 ,maximum1]) <= 0.9:
                continue
            if maximum0==max([maximum0 ,maximum1]):
                save_img(bboxes_f1,full_path)
                continue
            if maximum1==max([maximum0,maximum1]):
                save_img(bboxes_f2,full_path)
                continue


