# CS:GO Aimbot
*Note!* This aimbot was created as an exploration of machine learning technology. Using a computer to do your dirty work in an online competitive environment defeats the point, degrades the integrity of the game, and is completely unfulfilling. You're probably better than this aimbot, anyway. Significant performance improvements would result from running the convolutional layers on a GPU.  
[Google Slides Project Overview](https://docs.google.com/presentation/d/1JgeI7OwX7CPusGfQEWlTsAG5rQulyLh2D1QKVFxuxv8/edit?usp=sharing)  
## Setup
* Get the YOLOv3 trained weights [here](https://drive.google.com/file/d/19_mIm7uGL0IBHtYEjnRLX-X_7llLJWqE/view?usp=sharing) (34 MB)  
Put the weights at the top of the project directory.  
* You can install Counter-Strike: Global Offensive (CS:GO) for free [here](https://store.steampowered.com/app/730/CounterStrike_Global_Offensive/) 

   **Necessary In-Game Settings**  
  Set the window to *Windowed Fullscreen*  
  Turn *Raw Input OFF*  
  Turn *Mouse Acceleration OFF*  

* Install Anaconda3 for Python v3.7 [here](https://www.anaconda.com/distribution/)  
We highly suggest installing dependencies in a conda virtual environment.

```conda create --name opencv-env python=3.6  
conda activate opencv-env  
pip install opencv-contrib-python  
pip install imutils  
pip install numpy  
pip install cmake  
conda install -c conda-forge dlib=19.4  
pip install dlib --upgrade  
pip install Pillow  
pip install pyautogui  
pip install --upgrade mss
```
When finished,   
`conda deactivate`  
then  
`conda remove --name opencv-env --all`  
if you want to remove the virtual environment and everything in it.  
Use `conda list` to view installed modules  

## Running
```
usage: detector.py [-h] [-d] [-t TIME] [-r] [-f]  

optional arguments:  
  -h, --help            show this help message and exit  
  -d, --display         display bounding box images (default: False)  
  -t TIME, --time TIME  program run time in seconds (default: Eternity)  
  -r, --track           don't track detected objects (default: True)  
  -f, --fps             print frames-per-second (default: False)
```
