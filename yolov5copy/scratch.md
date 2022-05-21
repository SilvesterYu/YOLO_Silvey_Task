## Clues

```
python train.py --data coco128.yaml --cfg yolov5s.yaml --weights '' --batch-size 128 --epochs 1

```

```
python val.py --data coco128.yaml --weights yolov5s.pt --img 640 
```

train path: yolo_env/datasets/coco128/images/train2017 

val path: yolo_env/datasets/coco128/images/train2017

---

### detect.py

referenced class Annotator()

[Editing made]

(1) Drawing ellipse is done by modifying annotator.box_label() function in plots.py

---

### utils\plots.py: defined class Annotator()

Annotator() has self.draw, which comes from PIL import Image, ImageDraw, ImageFont, using ImageDraw.Draw(self.im)

self.im is an input from detect.py, which is im0. im0 comes from im0s.copy(). im0s comes from dataset. im0s coms from LoadStreams() from source

utils\dataloaders.py: defined LoadStreams() class which returns im0

Used Annotator().box_label() function. It # Add one xyxy box to image with label. It called cv2.rectangle() function, did something to self.im

annotator instance in detect.py returned annotator.result(), which is an np array of self.im. The code is "im0 = annotator.result()"

#### Todo: find out what exactly is im0

Annotator() has a rectangle() functions that adds  # Add rectangle to image (PIL-only)

utils\plots.py: plot_images() function references Annotator.rectangle(), which is used to annotate the result images

**plot_labels()** plots the labels and rectangles

plot_images(): the targets argument it takes is a 2-D array of 6 columns, column 0: perhaps means the image index column 1: class, column2-5: x y w h, column6: confidence that this item belongs to this class. It also calls annotator.rectangle(). When it annotates, it calls annotator.box_label() which is already modified.

[Editing made]

(1) plot_labels(): changed to ImageDraw.Draw(img).ellipse, because .rectangle() and .ellipse() both use upper-left + lower right coordinates as position points.

(2) Annotator.box_label(): it plots ellipse now, but the input is still upper-left + lower-right coordinates, which can be further modified to center point(x, y) and the 2 axes of an ellipse. There is the regular imageDraw.ellipse() and cv2.ellipse() which use different coordinates.

(3) plot_images(): calls plot_labels()

---

utils\plots.py uses *xywh2xyxy* function in plot_labels() function, plot_images(), 

utils\general.py defined *xywh2xyxy()* function, # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right


---

### Train.py

used plot_labels which calls *xywh2xyxy* function

labels is fed into plot_labels, labels come from labels = np.concatenate(dataset.labels, 0)

dataset comes from create_dataloader. train_path is fed. 

Train.py calls Model Object from models\yolo.py Model()

for calculating loss, it calls ComputeLoss() Object from loss.py

[Editing made]

(1) editing is made through plot_labels

---

### val.py

calls plot_images() from plots.py

---

### models/yolo.py

defined Model() class

m.anchors: still needs investigation

---

### loss.py

defined loss functions

defined class ComputeLoss() Object, which calls **bbox_iou** from utils.metrics.py: calculates the IoU loss


---

### utils/metrics.py

defined **bbox_iou()** which calculates the IoU loss



---
### Dataset Labels Format

Location: yolo_env/datasets/coco128/labels/

annotations for each image in form of a .txt file where each line of the text file describes a bounding box. 

- One row per object

- Each row is class x_center y_center width height format.

- Box coordinates must be normalized by the dimensions of the image (i.e. have values between 0 and 1)

- Class numbers are zero-indexed (start from 0).

![image](https://user-images.githubusercontent.com/74582280/169490740-a09251c7-0beb-4b13-81e8-16e833ebf397.png)


![image](https://user-images.githubusercontent.com/74582280/169490616-4ed91638-3b9c-4fc4-94be-df2282ac8a37.png)


---

hubconf.py: referenced AutoShape()

models\common.py: defined AutoShape()

---

## Current Progress

(1) May 20 - May 21: Edited annotation functions called in train.py, val.py, and detect.py so they know how to draw circles now, nice.

---

## Conclusions and must-dos

If we want to change the shape of anchor boxes, xywh2xyxy has to be modified

I need a training dataset. I'll then take the labeled boxes and first transform them into circles on the fly

The IoU loss function needs to be recalculated: **bbox_iou()** in utils/metrics.py
