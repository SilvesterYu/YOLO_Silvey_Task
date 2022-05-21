# YOLO_Silvey_Task
 Changing the bounding boxes from square to ellipse


### How to run

1. Setup the virtual envrionment on your computer

```
pip3 install virtualenv
virtualenv test
source test/bin/activate
```
2. Clone this repository
```
cd test
git clone https://github.com/SilvesterYu/YOLO_Silvey_Task.git
```

3. Run setup 
```
chmod -R 777 *.*
cd YOLO_Silvey_Task
chmod 0755 setup.sh
./setup.sh
```

**⚠️ IMPORTANT: Make sure that you have torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 exactly. You can check with**
```
pip3 list
```

## Testing

## 1. Test Train

Test train with

```
python train.py --data coco128.yaml --cfg yolov5s.yaml --weights '' --batch-size 128 --epochs 1
```

## 2. Test Val

Test val with

```
python val.py --data coco128.yaml --weights yolov5s.pt --img 640 
```

### 3. Test Detection
To test the detection, go into ``yolov5copy`` directory
```
cd yolov5copy
```

Then, get an image from internet
```
wget "https://a-z-animals.com/media/2021/12/Best-farm-animals-cow.jpg"
```

Run detection using
```
python3 detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source 'Best-farm-animals-cow.jpg' --save-txt
```
