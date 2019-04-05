from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
from PIL import ImageGrab, ImageDraw, Image, ImageColor
import time
import cv2
import pyautogui
import dlib

from input_mapper import InputMapper

# In CS:GO, turn on developer commands in settings, press '~',
# type sv_cheats 1, then bot_stop 1 to immobilize bots for easier testing.

class Detector:
    def __init__(self):
        self.args = self.get_args()

        # YOLOv3 vars
        self.colors = None  # Draw pretty bounding boxes
        self.classes = None  # Detectable objects
        self.yolo_config_path = 'yolov3.cfg'
        self.yolo_classes_path = 'yolov3.txt'
        self.yolo_weights_path = 'yolov3.weights'  # Pre-trained network
        self.net = cv2.dnn.readNet(self.yolo_weights_path, self.yolo_config_path)
        self.output_layers = self.get_output_layers(self.net)

        self.scale_down = .3
        self.scale_up = 1/self.scale_down
        self.nms_threshold = 0.4  # How eager are you to combine overlapping bounding boxes?
        self.conf_threshold = 0.5
        self.scale = 0.00392

        # PyAutoGUI
        self.im = InputMapper()

        # Tracker
        self.tracker = None
        self.max_track_time = 3  # seconds

        self.avg_fps = 0
        self.frames_captured = 0

        # Get classes from file
        with open(self.yolo_classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        #-----TOGGLE BY COMMENTING-----
        # Pass in display=True in detect_imgs() to display images.
        # If you've specified a path, it will save results to file.
        #self.detect_from_file()
        self.run()  # Ctrl + C failsafe

    def run(self):
        last_time = time.time() + 0.00001  # Non-zero
        total_time = 0
        time_diff = 0
        tracked_time = 0
        frames = 0

        while True:
            image = self.preprocess(self.im.screen_size)  # compressed img

            if self.tracker is None:
                outs = self.process(image)
                rd = self.find_target(image, outs)
                if rd:
                    target_pos = rd['center']  # Scaled up
                    rect = self.translate_rect(rd['rect'], target_pos)  # Scaled down
                    # Translate the tracking rect to where the crosshairs will be.
                    # This increases accuracy and prevents overshooting
                    self.im.move_mouse(target_pos)

                    rect = dlib.rectangle(rect[0], rect[1], rect[2], rect[3])
                    self.tracker.start_track(image, rect)
                    # time.sleep(.5)
            elif tracked_time > self.max_track_time:
                self.tracker = None
                tracked_time = 0
            else:
                # Update tracker and re-center mouse
                self.tracker.update(image)
                target_pos = self.tracker.get_position()
                center = dlib.center(target_pos)
                self.im.move_mouse(self.scale_point_up(center.x, center.y))
            tracked_time += time_diff

            # Print FPS log
            time_diff = time.time() - last_time
            total_time += time_diff
            frames += 1
            print('loop took {0:.3f} seconds at {1:.3f} fps. Avg fps: {2:.3f}'.format(
                time_diff, 1/time_diff, 1/(total_time/frames)))
            last_time = time.time()

    def translate_rect(self, rect, target_pos):
        #TODO: Adjust for scale
        dx, dy = self.im.distance_from_crosshairs(target_pos[0], target_pos[1])
        # DOn't want distance from each point... want distance from center point
        #xw, yh = self.im.distance_from_crosshairs(rect[2], rect[3])
        newr = (rect[0] + dx, rect[1] + dy, rect[2], rect[3])
        return newr

    def preprocess(self, screen_size):
        image = np.array(ImageGrab.grab(bbox=(0, 0, screen_size[0], screen_size[1])))
        image = imutils.resize(image, width=min(int(image.shape[1] * self.scale_down), image.shape[1]))
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def process(self, image):
        # Decreasing third parameter decreases computation time, incr false positives
        blob = cv2.dnn.blobFromImage(image, self.scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        return self.net.forward(self.output_layers)

    # Base code from Arun Punnusamy
    # https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html
    # Returns tuple of (center point, bounding box)
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
                    x = center_x - w / 2
                    y = center_y - h / 2

                    # DISPLAY IMAGE - testing
                    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 2)
                    cv2.imshow("detection", image)
                    cv2.waitKey(50)

                    # This will track the image in its scaled down size
                    # Create a new tracker object
                    self.tracker = dlib.correlation_tracker()
                    rect = (int(x), int(y), int(x + w), int(y + h))
                    return {'center': self.scale_point_up(center_x, center_y), 'rect': rect}
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

    def scale_point_up(self, x, y):
        return int(x * self.scale_up), int(y * self.scale_up)

    def scale_point_down(self, x, y):
        return int(x * self.scale_down), int(y * self.scale_down)

    # ------------NON-ESSENTIAL CODE------------
    def get_args(self):
        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--images", required=False, help="path to images directory")
        return vars(ap.parse_args())

    def save_img(self, filename, image):
        path = filename.split('\\')
        path = ".\\imgs\\results\\" + path[len(path) - 1]
        cv2.imwrite(path, image)

    # Some code inspired by Adrian Rosebrock
    # https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
    def detect_from_file(self, display=None):
        start_time = time.time()
        for image_path in paths.list_images(self.args["images"]):
            image = cv2.imread(image_path)
            image = imutils.resize(image, width=min(int(image.shape[1] * self.scale_down), image.shape[1]))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            width = image.shape[1]
            height = image.shape[0]

            # Decreasing third parameter decreases computation time, incr false positives
            blob = cv2.dnn.blobFromImage(image, self.scale, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            confidences = []
            boxes = []
            class_ids = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)

                    confidence = scores[class_id]
                    if confidence > 0.6 and self.classes[class_id] == 'person':
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w / 2
                        y = center_y - h / 2

                        # print("cx: {0} cy: {1}, w: {2} h: {3}, x: {4} y: {5}, a: {6} b: {7}".format
                        # (center_x, center_y, w, h, x, y, a, b))
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

            # Draw bounding boxes
            for i in indices:
                i = i[0]
                box = boxes[i]
                x = round(box[0])
                y = round(box[1])
                w = round(box[2])
                h = round(box[3])
                self.draw_prediction(image, class_ids[i], confidences[i], x, y, x + w, y + h)

            print("duration: {0:.2f}s".format(time.time() - start_time))

            filename = image_path[image_path.rfind("/") + 1:]
            self.save_img(filename, image)

            if display:
                cv2.imshow("After NMS", image)
                cv2.waitKey(2000)  # ms


if __name__ == "__main__":
    d = Detector()
