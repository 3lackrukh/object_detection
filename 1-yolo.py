#!/usr/bin/env python3
""" Module defines the class Yolo """
from tensorflow import keras as K
import numpy as np


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
                    outputs: number of outputs (predictions) made
                    anchor_boxes: number of anchor boxes for each prediction
                    2: [anchor_box_width, anchor_box_height]

        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes outputs from Darknet model for one image

        Arguments:
            outputs (list of numpy.ndarray):
                the predictions from the Darknet model for a single image:
                    Each output will have the shape (grid_height, grid_width,
                    anchor_boxes, 4 + 1 + classes)
                        grid_height: height of grid used for output
                        grid_width: width of grid used for output
                        anchor_boxes: number of anchor boxes used
                        4: (t_x, t_y, t_w, t_h)
                        1: box confidence
                        classes: class probabilities for all classes

            image_size (numpy.ndarray): array containing the image’s original
                size [image_height, image_width]

        Returns:
            tuple (boxes, box_confidences, box_class_probs):
                boxes: list of numpy.ndarray of shape (grid_height, grid_width,
                    anchor_boxes, 4) containing the processed boundary boxes
                    for each output, respectively:
                        4: (x1, y1, x2, y2)
                box_confidences: list of numpy.ndarray of shape (grid_height,
                    grid_width, anchor_boxes, 1) containing the processed box
                    confidences for each output, respectively
                box_class_probs: list of numpy.ndarray of shape (grid_height,
                    grid_width, anchor_boxes, classes) containing the processed
                    box class probabilities for each output, respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size

        for idx, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Create meshgrid for x and y coordinates
            cx = np.arange(grid_width).reshape(1, grid_width, 1)
            cy = np.arange(grid_height).reshape(grid_height, 1, 1)

            # Extract and process box coordinates
            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            # Apply sigmoid to tx, ty, and confidence
            bx = (1 / (1 + np.exp(-tx)) + cx) / grid_width
            by = (1 / (1 + np.exp(-ty)) + cy) / grid_height

            # Get anchor boxes for this output scale
            pw = self.anchors[idx, :, 0]
            ph = self.anchors[idx, :, 1]

            # Calculate width and height of boxes
            bw = pw * np.exp(tw) / self.model.input.shape[1]
            bh = ph * np.exp(th) / self.model.input.shape[2]

            # Calculate x1, y1, x2, y2
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            # Update boxes
            output[..., 0] = x1
            output[..., 1] = y1
            output[..., 2] = x2
            output[..., 3] = y2

            # Process confidences and class probabilities
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            boxes.append(output[..., :4])
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
