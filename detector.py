from __future__ import print_function
#from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
from PIL import ImageGrab
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
        self.display_fps = False
        self.tracking = True
        self.handle_args(args)
        print("This program will move and left-click your mouse. "
              "To exit, Alt-Tab to the command line and Ctrl-C.")
        print("Use [-h] for optional commands.")
        print("Beginning in 3 seconds. Happy hunting.")

        # Image cropping
        self.im = InputMapper()
        self.crop_scale = (.25, .3)  # x,y percent of image to remove (Must be below .5)
        ss = self.im.screen_size
        self.crop = (int(self.crop_scale[0] * ss[0]),
                     int(self.crop_scale[1] * ss[1]))  # x, y crop from all sides
        # Image scaling
        self.input_dim = 416  # YOLOv3 square pixel input size
        # resize factor = required YOLO input size / grabbed image width
        self.downsize_factor = self.input_dim / (ss[0] - self.crop[0] * 2)
        self.upsize_factor = 1 / self.downsize_factor  # inverse operation
        self.im.set_crop(self.crop)
        self.screen_size = self.im.screen_size

        # YOLOv3 vars
        self.classes = None  # Detectable objects
        self.yolo_classes_path = 'yolov3.txt'
        self.yolo_config_path = 'yolov3-tiny.cfg'
        self.yolo_weights_path = 'yolov3-tiny.weights'  # Pre-trained network in same directory.
        self.net = cv2.dnn.readNet(self.yolo_weights_path, self.yolo_config_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # Safely get classes from file
        with open(self.yolo_classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Tracker
        self.tracker = None
        self.max_track_time = 3  # seconds

        time.sleep(3)  # Breathing room before execution
        self.run()

    def run(self):
        """Main program loop."""
        last_time = time.time() + 0.00001  # Non-zero
        total_time = 0
        frames = 0
        tracked_time = 0

        # Main program loop
        while total_time < self.program_duration:
            # Update time
            time_diff = time.time() - last_time
            last_time = time.time()
            total_time += time_diff
            frames += 1

            image = self._preprocess()

            if self.tracker is None:  # If not currently tracking
                outs = self._process(image)
                rd = self.find_target(image, outs)  # Returns scaled down dict of center & BB rect
                if rd is not None:  # If target found, move reticle
                    target_pos = rd['center']
                    self.im.move_mouse(self.resize_point(target_pos, self.upsize_factor))

                    if self.tracking:  # Begin tracking if user has not specified otherwise.
                        rect = self._translate_rect(rd['rect'], target_pos)
                        self.tracker.start_track(dlib.as_grayscale(image), rect)
            elif tracked_time > self.max_track_time:
                self.tracker = None
                tracked_time = 0
            else:
                tracked_time *= self.track(image)
                tracked_time += time_diff

            if self.display_fps:  # Print FPS if specified by user.
                print('loop took {0:.3f} seconds at {1:.3f} fps. Avg fps: {2:.3f}'.format(
                    time_diff, 1/time_diff, 1/(total_time/frames)))

    def _track(self, image):
        """Once-per-frame tracker object update.
        Returns """
        # Update tracker
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        confidence = self.tracker.update(image)
        if confidence < 4.5:  # Stop tracking
            self.tracker = None
            return 0
        else:  # Re-center mouse
            target_pos = self.tracker.get_position()

            # Display bounding box if specified by user.
            if self.display_images:
                bl = dlib.drectangle.bl_corner(target_pos)
                tr = dlib.drectangle.tr_corner(target_pos)
                self.display_boxed_img(image, bl.x, bl.y, tr.x, tr.y)

            center = dlib.center(target_pos)
            self.im.move_mouse(self.resize_point((center.x, center.y), self.upsize_factor))
            return 1

    def _translate_rect(self, rect, target_pos):
        """Move the tracking rect to where it will be after the mouse has moved.
        Calculate the distance from the scaled down target to the center of the scaled down image."""
        scaled_down_center = self.resize_point((self.im.screen_centerx - self.crop[0],
                                               self.im.screen_centery - self.crop[1]),
                                               self.downsize_factor)
        dx = int(target_pos[0] - scaled_down_center[0])  # .shape is [y, x]
        dy = int(target_pos[1] - scaled_down_center[1])
        return dlib.translate_rect(rect, dlib.point(-dx, -dy))

    def _preprocess(self):
        """Return a cropped and resized portion of the user's screen. """
        image = np.array(ImageGrab.grab(bbox=(self.crop[0], self.crop[1],
                                              self.screen_size[0] - self.crop[0],
                                              self.screen_size[1] - self.crop[1])))

        return imutils.resize(image, width=min(int(image.shape[1] * self.downsize_factor), image.shape[1]))

    def _process(self, image):
        """Creates a blob and pass it into the network. """
        blob = cv2.dnn.blobFromImage(image,
                                     0.00392,  # 1 / 255. Do not tune
                                     (self.input_dim, self.input_dim), # Can be adjusted to 320, 416, 608
                                     (0, 0, 0),  #
                                     True)  # Convert to BGR for us
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        return outs

    # Base code from Arun Punnusamy
    # https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html
    def find_target(self, image, outs):
        """Scans network output and returns tuple of (center point, bounding box),
        relative to cropped & scaled down image."""
        t = time.time()
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.55 and self.classes[class_id] == 'person':
                    # Get bounding box for tracking and drawing, center for shooting
                    width = image.shape[1]
                    height = image.shape[0]
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Display image with BB if specified by user
                    if self.display_images:
                        self.display_boxed_img(image, x, y, x + w, y + h)

                    # Create a new tracker object to begin tracking
                    if self.tracking:
                        self.tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x, y, x + w, y + h)
                    print("Process Elapsed: ", time.time() - t)
                    return {'center': (center_x, center_y), 'rect': rect}
        return None

    def handle_args(self, args):
        """Sets instance variables from get_args() command line output."""
        if args["time"] is not None:
            self.program_duration = int(args["time"])
        else:
            self.program_duration = math.inf
        self.display_images = args["display"]
        self.tracking = args["track"]
        self.display_fps = args["fps"]

    @staticmethod
    def resize_point(point, factor):
        """Returns a tuple of x and y scaled up."""
        return int(point[0] * factor), int(point[1] * factor)

    @staticmethod
    def get_args():
        """Parses command line arguments on program execution."""
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--display", required=False, action="store_true",
                        help="display bounding box images (default: False)")
        ap.add_argument("-t", "--time", required=False, help="program run time in seconds (default: Eternity)")
        ap.add_argument("-r", "--track", required=False, action="store_false",
                        help="don't track detected objects (default: True)")
        ap.add_argument("-f", "--fps", required=False, action="store_true",
                        help="print frames-per-second")
        return vars(ap.parse_args())

    @staticmethod
    def display_boxed_img(image, x, y, x_plus_w, y_plus_h):
        """Displays the image with a white bounding box drawn on it."""
        cv2.rectangle(image, (int(x), int(y)), (int(x_plus_w), int(y_plus_h)), (255, 255, 255), 2)
        cv2.imshow("detection", image)
        cv2.waitKey(10)


if __name__ == "__main__":
    Detector()
