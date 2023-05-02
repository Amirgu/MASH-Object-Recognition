from DiffusionDet.diffusiondet import predictor
from DiffusionDet import demo
from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2


def setup_cfg():
    # load config from file and command-line arguments
    cfg = demo.get_cfg()
    demo.add_diffusiondet_config(cfg)
    demo.add_model_ema_configs(cfg)
    cfg.merge_from_file('/content/DiffusionDet/configs/diffdet.coco.res50.yaml')
    cfg.MODEL.WEIGHTS = '/content/DiffusionDet/diffdet_coco_res50.pth'

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.freeze()

    return cfg


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=1,
          font_thickness=2,
          text_color=(255, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)



config = setup_cfg()

# Run multi-object tracker with diffusiondet
def run_MOT_DD(folder_img_path, tracking_txt_path, output_path, config):

  # Names of categories for classification
  COCO_INSTANCE_CATEGORY_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

  # Initialize the centroid tracker
  ct = CentroidTracker()

  VizDemo = demo.VisualizationDemo(config)

  file = open(tracking_txt_path, "w")

  number_frames = len(fnmatch.filter(os.listdir(folder_img_path ), '*.*'))


  # Visit the frames
  for i in range(1,number_frames + 1):
    frame = '{}'.format(i)

    #To be compatible with the MOT17 dataset images names
    while len(frame)<6:
      frame = '0' + frame
    filename = folder_img_path + frame + '.jpg'

    # Read the frame
    img = demo.read_image(filename)

    # Detect objects
    predictions, visualized_output  = VizDemo.run_on_image(img)
    predictions =  predictions['instances']

    # Get boxes coordinates
    box = predictions._fields['pred_boxes'].tensor.cpu().numpy().astype(int)
    
    # Get classes and scores
    scores= predictions._fields['scores'].cpu().numpy()
    clas = predictions._fields['pred_classes'].cpu().numpy()

    # Put boxes coordinates in a tuple
    result = tuple(map(tuple, box))

    # Put rectangles on the frame and write the class
    for j in range(len(result)):
      cv2.rectangle(img,result[j][0:2],result[j][2:4],(255, 0, 0),4)
      draw_text(img, COCO_INSTANCE_CATEGORY_NAMES[clas[j]], pos= (result[j][0],result[j][1]))

    # Track objects and write the IDs objects on the frame
    result = np.array(result)
    objects, tab = ct.update(result)
    for (objectID, centroid) in objects.items():
      text = "ID {}".format(objectID)
      cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Write the file with the predictions boxes
    for k in range(len(tab)):
      file.write(str(i) + ',' + str(tab[k][0]) + ',' + str(tab[k][1]) + ',' + str(tab[k][2]) +  ',' + str(tab[k][3]) + ',' + str(tab[k][4]) + '\n' )
   
    # register the output frame
    cv2.imwrite(output_path+filename[-10:], img)

  file.close()