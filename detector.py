from __future__ import print_function
#from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
from PIL import ImageGrab, ImageDraw, Image, ImageColor
import time
import cv2
import dlib
import math

from input_mapper import InputMapper

# In CS:GO, turn on developer commands in settings, press '~',
# type sv_cheats 1, then bot_stop 1 to immobilize bots for easier testing.

class Detector:
    def __init__(self):
        args = self.get_args()
        self.program_duration = 0
        self.display_images = False
        self.tracking = True
        self.handle_args(args)
        print("This program will move and left-click your mouse. "
              "To exit, Alt-Tab to the command line and Ctrl-C.")
        print("Use [-h] for optional commands.")
        print("Happy hunting")

        # Cropping & Scaling PyAutoGUI
        self.im = InputMapper()
        self.crop_scale = (.25, .33)  # x,y percent of image to remove (Must be below .5)
        ss = self.im.screen_size
        self.crop = (int(self.crop_scale[0] * ss[0]),
                     int(self.crop_scale[1] * ss[1]))  # x, y crop from all sides
        self.scale_down = .28  # Image resize factor
        self.scale_up = 1 / self.scale_down
        self.im.set_crop(self.crop)

        self.avg_fps = 0
        self.frames_captured = 0

        # YOLOv3 vars
        self.colors = None  # Draw pretty bounding boxes
        self.classes = None  # Detectable objects
        self.yolo_config_path = 'yolov3.cfg'
        self.yolo_classes_path = 'yolov3.txt'
        self.yolo_weights_path = 'yolov3.weights'  # Pre-trained network
        self.net = cv2.dnn.readNet(self.yolo_weights_path, self.yolo_config_path)
        self.output_layers = self.get_output_layers(self.net)

        # Tracker
        self.tracker = None
        self.max_track_time = 3  # seconds

        # Get classes from file
        with open(self.yolo_classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        #self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        time.sleep(3)
        self.run()

    def run(self):
        last_time = time.time() + 0.00001  # Non-zero
        total_time = 0
        time_diff = 0
        tracked_time = 0
        frames = 0

        while total_time < self.program_duration:
            image = self.preprocess(self.im.screen_size)  # compressed img

            if self.tracker is None:
                outs = self.process(image)
                rd = self.find_target(image, outs)  # Returns scaled down values!
                if rd is not None:  # If target found, move and track
                    target_pos = rd['center']
                    self.im.move_mouse(self.scale_point_up(target_pos[0], target_pos[1]))

                    if self.tracking:
                        rect = self.translate_rect(rd['rect'], target_pos)
                        self.tracker.start_track(dlib.as_grayscale(image), rect)
            elif tracked_time > self.max_track_time:
                self.tracker = None
                tracked_time = 0
            else:
                tracked_time *= self.track(image)
            tracked_time += time_diff

            # Print FPS
            time_diff = time.time() - last_time
            total_time += time_diff
            frames += 1
            print('loop took {0:.3f} seconds at {1:.3f} fps. Avg fps: {2:.3f}'.format(
                time_diff, 1/time_diff, 1/(total_time/frames)))
            last_time = time.time()

    def track(self, image):
        # Update tracker and re-center mouse
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        confidence = self.tracker.update(image)
        # print("TRACKING CONFIDENCE: ", confidence)
        if confidence < 4.5:
            self.tracker = None
            return 0
        else:
            target_pos = self.tracker.get_position()

            # DRAW TEST
            if self.display_images:
                bl = dlib.drectangle.bl_corner(target_pos)
                tr = dlib.drectangle.tr_corner(target_pos)
                self.display_boxed_img(image, bl.x, bl.y, tr.x, tr.y)

            center = dlib.center(target_pos)
            self.im.move_mouse(self.scale_point_up(center.x, center.y))
            return 1

    # Move the tracking rect to where it will be after the mouse has moved. Calculate the distance
    # from the scaled down target to the center of the scaled down image.
    def translate_rect(self, rect, target_pos):
        scaled_down_center = self.scale_point_down(self.im.screen_centerx - self.crop[0],
                                                   self.im.screen_centery - self.crop[1])
        dx = int(target_pos[0] - scaled_down_center[0])  # .shape is y | x
        dy = int(target_pos[1] - scaled_down_center[1])
        return dlib.translate_rect(rect, dlib.point(-dx, -dy))

    def preprocess(self, screen_size):
        image = np.array(ImageGrab.grab(bbox=(self.crop[0], self.crop[1],
                                              screen_size[0] - self.crop[0],
                                              screen_size[1] - self.crop[1])))
        image = imutils.resize(image, width=min(int(image.shape[1] * self.scale_down), image.shape[1]))
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def process(self, image):
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        return self.net.forward(self.output_layers)

    # Base code from Arun Punnusamy
    # https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html
    # Returns tuple of (center point, bounding box), relative to cropped & scaled down image
    def find_target(self, image, outs):
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.6 and self.classes[class_id] == 'person':
                    print("DETECTION")
                    # Get bounding box for tracking and drawing, center for shooting
                    width = image.shape[1]
                    height = image.shape[0]
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # DISPLAY IMAGE - testing
                    if self.display_images:
                        self.display_boxed_img(image, x, y, x + w, y + h)

                    # This will track the image in its scaled down size
                    # Create a new tracker object
                    if self.tracking:
                        self.tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x, y, x + w, y + h)
                    return {'center': (center_x, center_y), 'rect': rect}
        return None

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = '{0}, {1:.2f}'.format(self.classes[class_id], confidence)
        color = self.colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def display_boxed_img(self, image, x, y, x_plus_w, y_plus_h):
        cv2.rectangle(image, (int(x), int(y)), (int(x_plus_w), int(y_plus_h)), (255, 255, 255), 2)
        cv2.imshow("detection", image)
        cv2.waitKey(10)

    def scale_point_up(self, x, y):
        return int(x * self.scale_up), int(y * self.scale_up)

    def scale_point_down(self, x, y):
        return int(x * self.scale_down), int(y * self.scale_down)

    def get_args(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--display", required=False, action="store_true",
                        help="display bounding box images (default: False)")
        ap.add_argument("-t", "--time", required=False, help="program run time in seconds (default: Eternity)")
        ap.add_argument("-r", "--track", required=False, action="store_false",
                        help="don't track detected objects (default: True)")
        return vars(ap.parse_args())

    def handle_args(self, args):
        if args["time"] is not None:
            self.program_duration = int(args["time"])
        else:
            self.program_duration = math.inf

        self.display_images = args["display"]
        self.tracking = args["track"]


if __name__ == "__main__":
    d = Detector()
