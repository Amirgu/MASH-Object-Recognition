# MASH-Object-Recognition

School projects in the Object Recognition and Computer Vision Course (I. LAPTEV, J. PONCE, C. SCHMID, J. SIVIC) shared between Master 2 MASH in Paris Dauphine University 
and Master 2 MVA in ENS


### Kaggle Competition: Bird Classification

MVA Kaggle Competition : Bird Classification  (https://www.kaggle.com/competitions/mva-recvis-2022), Our solution got 84% Accuracy on the final test set.
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis/assignment3/bird_dataset.zip). The test image labels are not provided.


#### Cropping 

Run the file crop.py

#### Data augmentation 

Run The file augementation.py

#### Training

Run main.py

#### Evalution

Run eval.py

### Object Detection and Tracking : DiffusionDet

We studied DiffusionDet which is a diffusion model for object detection. Specifically, we reproduced the results of the author on MS-COCO dataset and compared DiffusionDet performances with Faster R-CNN. The main goal was to extend the use of Diffusion model to Multi-Object Tracking. We implemented a centroid-based tracker on top of the DiffusionDet model.

#### Results



https://user-images.githubusercontent.com/127419134/235793017-8e57ba41-a68f-4296-9d3d-820bc148a957.mp4


#### Datasets

We have used the [MS-COCO dataset]() for our object detection experiments and the [MOT17 dataset]() for our Multi-Object Tracking experiments.

#### Bibliography

Chen & al., *DiffusionDet : Diffusion Model for Object Detection* ([arxiv](https://arxiv.org/abs/2211.09788))

```
@misc{https://doi.org/10.48550/arxiv.2211.09788,
  doi = {10.48550/ARXIV.2211.09788},
  
  url = {https://arxiv.org/abs/2211.09788},
  
  author = {Chen, Shoufa and Sun, Peize and Song, Yibing and Luo, Ping},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {DiffusionDet: Diffusion Model for Object Detection},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
