### Copy of yolov5 for programming practice. Cloned on May 16, 2022. Not my original work.

# Original repo: https://github.com/ultralytics/yolov5

### Downgrade the YOLO from GPU to CPU

```
pip3 install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

When running on mac, you will need torch 1.10 (torch 1.11 is unstable)

```
pip3 install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1

```

### To Run

Execute the following outside of yolo_env folder

```
sudo -s
```

Activate yolo environment

```
source yolo_env/bin/activate
```

Then cd into yolo_env/yolov5

```
cd  yolo_env/yolov5copy
```

Download a custom image from the internet 

```
wget "https://a-z-animals.com/media/2021/12/Best-farm-animals-cow.jpg"
```

To allow access

```
sudo chmod -R 777 *.*
```

Then, run yolo detection with

```
python3 detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source 'Best-farm-animals-cow.jpg' --save-txt
```
 
Use tiv to display the saved image in terminal

```
feh runs/detect/exp3/Best-farm-animals-cow.jpg
```
Terminal image viewer: https://github.com/stefanhaustein/TerminalImageViewer

Train:
```
python train.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
```
```
python train.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                       yolov5s                                64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```


