#!/usr/bin/env python3
""" Module defines the class Yolo """
from tensorflow import keras as K
import numpy as np
import os
import cv2


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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on a threshold
        Arguments:
            boxes (list of numpy.ndarray):
                list of numpy.ndarray of shape (grid_height, grid_width,
                    anchor_boxes, 4) containing the processed boundary boxes
                    for each output, respectively:
                        4: (x1, y1, x2, y2)
            box_confidences (list of numpy.ndarray):
                list of numpy.ndarray of shape (grid_height, grid_width,
                    anchor_boxes, 1) containing the processed box confidences
                    for each output, respectively
            box_class_probs (list of numpy.ndarray):
                list of numpy.ndarray of shape (grid_height, grid_width,
                    anchor_boxes, classes) containing the processed box class
                    probabilities for each output, respectively

        Returns:
            tuple (filtered_boxes, box_classes, box_scores):
                filtered_boxes: numpy.ndarray of shape (?, 4) containing all of
                    the filtered bounding boxes:
                box_classes: numpy.ndarray of shape (?,) containing the class
                    number for each box in filtered_boxes
                box_scores: numpy.ndarray of shape (?) containing the box
                    scores for each box in filtered_boxes
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []
        for i in range(len(boxes)):
            # Get dimensions for reshaping
            grid_h, grid_w, anchors, _ = boxes[i].shape
            # Reshape box_confidences: (grid_h * grid_w * anchors, 1)
            confidences = box_confidences[i].reshape(-1, 1)
            # Reshape box_class_probs: (grid_h * grid_w * anchors, classes)
            class_probs = box_class_probs[i].reshape(
                -1, box_class_probs[i].shape[-1])

            # Multiply confidences by class probabilities
            scores = confidences * class_probs
            # Get max score and corresponding class for each box
            max_scores = np.max(scores, axis=1)
            class_indices = np.argmax(scores, axis=1)

            # Find indices of boxes with max score greater than threshold
            mask = max_scores >= self.class_t

            if len(mask) > 0:
                # Reshape boxes: (grid_h * grid_w * anchors, 4)
                current_boxes = boxes[i].reshape(-1, 4)
                # Add filtered results to output lists
                filtered_boxes.append(current_boxes[mask])
                box_classes.append(class_indices[mask])
                box_scores.append(max_scores[mask])

        if filtered_boxes:
            filtered_boxes = np.concatenate(filtered_boxes, axis=0)
            box_classes = np.concatenate(box_classes, axis=0)
            box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-max suppression to filtered boxes

        Arguments:
            filtered_boxes (numpy.ndarray):
                numpy.ndarray of shape (?, 4) containing all of the filtered
                bounding boxes:
            box_classes (numpy.ndarray):
                numpy.ndarray of shape (?,) containing the class number for
                each box in filtered_boxes
            box_scores (numpy.ndarray):
                numpy.ndarray of shape (?) containing the box scores for each
                box in filtered_boxes

        Returns:
            tuple containing:
                box_predictions:
                    numpy.ndarray of shape (?, 4) containing all of
                    the predicted bounding boxes ordered by class and box score
                predicted_box_classes:
                    numpy.ndarray of shape (?) containing the class number
                    for each box in box_predictions
                predicted_box_scores:
                    numpy.ndarray of shape (?) containing the box scores
                    for each box in box_predictions
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Get unique classes
        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            # Get indices of boxes for current class
            indices = np.where(box_classes == cls)[0]

            # Get boxes and scores for current class
            class_boxes = filtered_boxes[indices]
            class_scores = box_scores[indices]

            # Sort by score in descending order
            score_sort = np.argsort(-class_scores)
            class_boxes = class_boxes[score_sort]
            class_scores = class_scores[score_sort]

            while len(class_boxes) > 0:
                # Take the box with highest score
                box_predictions.append(class_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(class_scores[0])

                if len(class_boxes) == 1:
                    break

                # Compare with rest of the boxes
                box = class_boxes[0]

                # Calculate coordinates of intersection
                x1 = np.maximum(box[0], class_boxes[1:, 0])
                y1 = np.maximum(box[1], class_boxes[1:, 1])
                x2 = np.minimum(box[2], class_boxes[1:, 2])
                y2 = np.minimum(box[3], class_boxes[1:, 3])

                # Calculate intersection area
                intersection_area = np.maximum(0, x2 - x1) * \
                    np.maximum(0, y2 - y1)

                # Calculate union area
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                class_boxes_area = (class_boxes[1:, 2] - class_boxes[1:, 0]) *\
                    (class_boxes[1:, 3] - class_boxes[1:, 1])
                union_area = box_area + class_boxes_area - intersection_area

                # Calculate IoU
                iou = intersection_area / union_area

                # Keep boxes with IoU less than threshold
                keep_indices = np.where(iou < self.nms_t)[0]
                class_boxes = class_boxes[keep_indices + 1]
                class_scores = class_scores[keep_indices + 1]

        # Convert results to numpy arrays
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    def load_images(self, folder_path):
        """
        Loads images from a folder

        Arguments:
            folder_path (str): path to the folder containing the images

        Returns:
            tuple containing:
                images (list):
                    list of images as numpy arrays
                image_paths (list):
                    list of paths to the images
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                images.append(image)
                image_paths.append(image_path)

        return images, image_paths
