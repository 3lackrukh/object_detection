#!/usr/bin/env python3
""" Module defines the lass Yolo """
from tensorflow import keras as K


class Yolo:
    """ Class implements the Yolo v3 algorithm to perform object detection """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor

        Arguments:
            model_path (str): path to where Darknet Keras model is stored

            classes_path (str): path to list of class names in order of index
                used for the model

            class_t (float): box score threshold to filter boxes by

            nms_t (float): IOU threshold for non-max suppression

            anchors (numpy.ndarray): array of shape (outputs, anchor_boxes, 2)
                containing all anchor boxes:
                    outputs: number of outputs (predictions) made by the YOLO model
                    anchor_boxes: number of anchor boxes used for each prediction
                    2: [anchor_box_width, anchor_box_height]

        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors