# Human pose estimation and classification using Torchivsion

This programm classifies poses (walk, fall, fallen, sitting) using Mediapipe for human pose estimation. This programm prototype can classify several persons in one frame. With 37 training videos and 11 test videos it showed about 88% of accuracy on classyfing fall, fallen and walking poses and about 10 FPS on RTX 3060 12GB and i5 12400F.

Example of work: 



https://github.com/Kexitor/HPE_Torchvision/assets/55799671/21bdbc9e-70a9-462b-b04a-1c55ead0612e

https://youtu.be/_BdMVwT1hZ4

https://www.youtube.com/watch?v=uedp3CnXWmM

<!--## To launch this YOLOv7 required:
```
git clone https://github.com/WongKinYiu/yolov7.git
```

Firtsly it is better to clone YOLOv7 by upper link and install requirements.txt and after that overwrite that cloned repo with my files.

Or just install requirements by this command:

```
pip install -r requirements.txt
```

This file needed in main directory: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt-->

It is better to use the GPU to work with a program prototype.

## Used videos for training and testing:

https://www.youtube.com/watch?v=8Rhimam6FgQ

http://fenix.ur.edu.pl/mkepski/ds/uf.html

## Used lib versions:

Python==3.7.8

matplotlib==3.5.3

numpy==1.21.6

opencv-python==4.6.0.66

Pillow==9.3.0

scipy==1.7.3

torch==1.13.1+cu117

torchaudio==0.13.1+cu116

torchvision==0.14.1+cu117

pandas==1.3.5

scikit-learn==1.0.2



## How to use:

`data_generator.py` used to make data for training. Also by this file is used `data_lists.py` and numerous of videos to generate CSV file data. All params for generating CSV are in code.


Example of usage:
```
python data_generator.py
```
`keypoint_rcnn.py` is used to classify pose of person on video. This file uses CSV generated by previos command. In current version file `pm_37vtrain_tv.pkl` already contains training data. If you want train model by yourself uncomment lines 29-30. Training by using MLPClassifier can take long time depending on dataset size and CPU. Data paths can be changed in code.

Example of usage:
```
python keypoint_rcnn.py -i videos\50wtf.mp4
```


You can check my other prototypes: 

https://github.com/Kexitor/HPE_Mediapipe

https://github.com/Kexitor/HPE_YOLOv7
