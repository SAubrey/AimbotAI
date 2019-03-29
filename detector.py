from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
from PIL import ImageGrab
import time
import cv2


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

        self.nms_threshold = 0.4  # How eager are you to combine overlapping bounding boxes?
        self.conf_threshold = 0.5
        self.scale = 0.00392

        # Get classes from file
        with open(self.yolo_classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        #-----COMMENT OUT WHAT YOU DON'T WANT-----
        # Pass in display=True in detect_imgs() to display images.
        # If you've specified a path, it will save results to file.
        self.detect_imgs()
        #self.record_screen()  # 'q' to stop - Doesn't always work? Ctrl + C failsafe.

    def get_args(self):
        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--images", required=True, help="path to images directory")
        return vars(ap.parse_args())

    def detect_imgs(self):
        # Some code inspired by Adrian Rosebrock
        # https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
        start_time = time.time()
        # loop over the image paths
        for image_path in paths.list_images(self.args["images"]):
            image = cv2.imread(image_path)
            self.detect(image, image_path)

        print("duration: {0:.2f}s".format(time.time() - start_time))


    # Base code taken from Arun Punnusamy
    # https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html
    def detect(self, image, image_path=None, display=False):
        image = imutils.resize(image, width=min(500, image.shape[1])) # Higher res == better detection?
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        Width = image.shape[1]
        Height = image.shape[0]

        confidences = []
        boxes = []
        class_ids = []

        # Decreasing third parameter decreases computation time, incr false positives
        blob = cv2.dnn.blobFromImage(image, self.scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers(self.net))

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)

                confidence = scores[class_id]
                if confidence > 0.5 and self.classes[class_id] == 'person':
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
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

        if image_path:
            filename = image_path[image_path.rfind("/") + 1:]
            self.save_img(filename, image)
        if display:
            cv2.imshow("After NMS", image)
            cv2.waitKey(2500)  # ms

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = '{0}, {1:.2f}'.format(self.classes[class_id], confidence)
        color = self.colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def record_screen(self):
        last_time = time.time()
        while True:
            # 800x600 windowed mode
            printscreen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640))) #LOOKUP
            time_diff = time.time() - last_time
            print('loop took {0:.3f} seconds at {1:.3f} fps'.format(time_diff, .6/time_diff))
            last_time = time.time()

            #printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
            self.detect(printscreen)
            cv2.imshow('window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    def save_img(self, filename, image):
        path = filename.split('\\')
        path = ".\\imgs\\results\\" + path[len(path) - 1]
        cv2.imwrite(path, image)
    """
    # HOG + SVM detector built in to openCV - not very accurate detection
    def init_hog(self):
        # initialize the HOG descriptor/person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return hog


    def detect1(self, image, image_path=None, display=False):

        # detect people in the image
        (rects, weights) = self.hog.detectMultiScale(image, winStride=(4, 4),
                                                padding=(8, 8), scale=1.07)  # Tune scale/winStride

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        """


if __name__ == "__main__":
    d = Detector()
