# CS:GO Aimbot
## Setup
Get the YOLOv3 trained weights [here](https://drive.google.com/file/d/1tZ_bju3p52w00HpYJPz6xbhbiMEQPkwS/view?usp=sharing)  
(246 MB)  

Install Anaconda3  

```conda create --name opencv-env python=3.6  
conda activate opencv-env  
pip install opencv-contrib-python  
pip install imutils  
pip install numpy  
pip install cmake  
conda install -c conda-forge dlib=19.4  
pip install Pillow  
pip install pyautogui  
```
When finished,   
`conda deactivate`   
or  
`conda remove --name opencv-env --all`  

Use `conda list` to view installed modules  

## Running
```usage: detector.py [-h] [-d] [-t TIME] [-r]  

optional arguments:  
  -h, --help            show this help message and exit  
  -d, --display         display bounding box images (default: False)  
  -t TIME, --time TIME  program run time in seconds (default: Eternity)  
  -r, --track           don't track detected objects (default: True)```  

