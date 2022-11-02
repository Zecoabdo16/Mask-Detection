"""
Many thanks to [This notebook](https://www.kaggle.com/drjerk/detect-faces-using-yolo/#data) on kaggle
"""

# Commented out IPython magic to ensure Python compatibility.
# Model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Conv2D, Input, ZeroPadding2D
from keras.models import Model

# For modifying the images
import numpy as np
import cv2

# For visualization purposes
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# %matplotlib inline

# Some helping functions
def transpose_shots(shots):
    return [(shot[1], shot[0], shot[3], shot[2], shot[4]) for shot in shots]

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def non_max_suppression(boxes, prob, threshold):

    if len(boxes) == 0:
        return np.array([])
    
    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    indexes = np.argsort(prob)
    true_boxes_indexes = []

    while len(indexes) > 0:
        true_boxes_indexes.append(indexes[-1])

        w_inter = np.maximum(np.minimum(x2[indexes[:-1]], x2[indexes[-1]]) - np.maximum(x1[indexes[:-1]], x1[indexes[-1]]), 0)
        h_inter = np.maximum(np.minimum(y2[indexes[:-1]], y2[indexes[-1]]) - np.maximum(y1[indexes[:-1]], y1[indexes[-1]]), 0)
        intersection = w_inter * h_inter          # Area of intersection
        
        area_rest = (x2[indexes[:-1]] - x1[indexes[:-1]]) * (y2[indexes[:-1]] - y1[indexes[:-1]])
        area_last = (x2[indexes[-1]] - x1[indexes[-1]]) * (y2[indexes[-1]] - y1[indexes[-1]])

        iou = intersection / (area_rest +  area_last - intersection)

        indexes = np.delete(indexes, -1)
        indexes = np.delete(indexes, np.where(iou >= threshold)[0])

    return boxes[true_boxes_indexes]

def union_suppression(boxes, threshold):
    """
    To eliminate redundant boxes over different shots.
    """
    if len(boxes) == 0:
        return np.array([])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    indexes = np.argsort((x2 - x1) * (y2 - y1))
    result_boxes = []

    while len(indexes) > 0:
        w_inter = np.maximum(np.minimum(x2[indexes[:-1]], x2[indexes[-1]]) - np.maximum(x1[indexes[:-1]], x1[indexes[-1]]), 0)
        h_inter = np.maximum(np.minimum(y2[indexes[:-1]], y2[indexes[-1]]) - np.maximum(y1[indexes[:-1]], y1[indexes[-1]]), 0)
        intersection = w_inter * h_inter          # Area of intersection

        area_rest = (x2[indexes[:-1]] - x1[indexes[:-1]]) * (y2[indexes[:-1]] - y1[indexes[:-1]])
        area_last = (x2[indexes[-1]] - x1[indexes[-1]]) * (y2[indexes[-1]] - y1[indexes[-1]])
        min_s = np.minimum(area_rest, area_last)

        ioms = intersection / (min_s + 1e-9)
        neighbours = np.where(ioms >= threshold)[0]

        if len(neighbours) > 0:
            x1_result = min(np.min(x1[indexes[neighbours]]), x1[indexes[-1]])
            y1_result = min(np.min(y1[indexes[neighbours]]), y1[indexes[-1]])
            x2_result = max(np.max(x2[indexes[neighbours]]), x2[indexes[-1]])
            y2_result = max(np.max(y2[indexes[neighbours]]), y2[indexes[-1]])
            result_boxes.append([x1_result, y1_result, x2_result, y2_result])
        else:
            result_boxes.append([x1[indexes[-1]], y1[indexes[-1]], x2[indexes[-1]], y2[indexes[-1]]])

        indexes = np.delete(indexes, -1)
        indexes = np.delete(indexes, neighbours)

    return result_boxes

# model loading
def load_mobilenetv2_224_075_detector(path):
    input_tensor = Input(shape=(224, 224, 3))
    output_tensor = MobileNetV2(weights=None, include_top=False, input_tensor=input_tensor, alpha=0.75).output
    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = Conv2D(kernel_size=(3, 3), filters=5)(output_tensor)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.load_weights(path)
    
    return model

mobilenetv2 = load_mobilenetv2_224_075_detector(r"facedetection-mobilenetv2-size224-alpha0.75.h5")

class FaceDetector():
    """  
    __init__ parameters:
    -------------------------------
    model - model to infer
    shots - list of aspect ratios that images could be (described earlier)
    image_size - model's input size (hardcoded for mobilenetv2)
    grids - model's output size (hardcoded for mobilenetv2)
    union_threshold - threshold for union of predicted boxes within multiple shots
    iou_threshold - IOU threshold for non maximum suppression used to merge YOLO detected boxes for one shot,
                    you do need to change this because there are one face per image as I can see from the samples
    prob_threshold - probability threshold for YOLO algorithm, you can balance beetween precision and recall using this threshold
    
    returns:
    -------------------------------
    list of 4 element tuples (left corner x, left corner y, right corner x, right corner y) of detected boxes within [0, 1] range
    """

    # Different shots taken for each image
    SHOTS = {
        'aspect_ratio' : 16/9,
        'shots' : [
             (0, 0, 9/16, 1, 1),
             (7/16, 0, 9/16, 1, 1),
             (0, 0, 5/16, 5/9, 0.5),
             (0, 4/9, 5/16, 5/9, 0.5),
             (11/48, 0, 5/16, 5/9, 0.5),
             (11/48, 4/9, 5/16, 5/9, 0.5),
             (22/48, 0, 5/16, 5/9, 0.5),
             (22/48, 4/9, 5/16, 5/9, 0.5),
             (11/16, 0, 5/16, 5/9, 0.5),
             (11/16, 4/9, 5/16, 5/9, 0.5),
        ]
    }
    
    SHOTS_T = {
        'aspect_ratio' : 9/16,
        'shots' : transpose_shots(SHOTS['shots'])
    } 

    def __init__(self, model=mobilenetv2, shots=[SHOTS, SHOTS_T], image_size=224, grids=7, iou_threshold=0.1, union_threshold=0.25, prob_threshold=0.5):
        self.model = model
        self.shots = shots
        self.image_size = image_size
        self.grids = grids
        self.iou_threshold = iou_threshold
        self.union_threshold = union_threshold
        self.prob_threshold = prob_threshold
        
    
    def detect(self, frame):     
        # Making sure that the aspect ratio is proper 
        if abs(frame.shape[1] / frame.shape[0] - 16/9) < abs(frame.shape[1] / frame.shape[0] - 9/16):
            frame = cv2.resize(frame,(1600,900)) 
            shots = self.shots[0]
            
        else:
            frame = cv2.resize(frame,(900,1600)) 
            shots = self.shots[1]
            

        frames = []  # taking different shots
        for s in shots["shots"]:
            frames.append(cv2.resize(frame[round(s[1] * frame.shape[0]):round((s[1] + s[3]) * frame.shape[0]), 
                                           round(s[0] * frame.shape[1]):round((s[0] + s[2]) * frame.shape[1])], 
                                     (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST))
        frames = np.array(frames)

        predictions = self.model.predict(frames)  

        # Converting the output of the model into coordinates
        boxes = []
        prob = []
        shots = shots['shots']
        for i in range(len(shots)):
            slice_boxes = []
            slice_prob = []
            for j in range(predictions.shape[1]):
                for k in range(predictions.shape[2]):
                    p = sigmoid(predictions[i][j][k][4])
                    if not(p is None) and p > self.prob_threshold:
                        px = sigmoid(predictions[i][j][k][0])
                        py = sigmoid(predictions[i][j][k][1])
                        pw = min(np.exp(predictions[i][j][k][2] / self.grids), self.grids)
                        ph = min(np.exp(predictions[i][j][k][3] / self.grids), self.grids)
                        if not(px is None) and not(py is None) and not(pw is None) and not(ph is None) and pw > 1e-9 and ph > 1e-9:
                            cx = (px + j) / self.grids
                            cy = (py + k) / self.grids
                            wx = pw / self.grids
                            wy = ph / self.grids
                            if wx <= shots[i][4] and wy <= shots[i][4]:
                                lx = min(max(cx - wx / 2, 0), 1)
                                ly = min(max(cy - wy / 2, 0), 1)
                                rx = min(max(cx + wx / 2, 0), 1)
                                ry = min(max(cy + wy / 2, 0), 1)

                                lx *= shots[i][2]
                                ly *= shots[i][3]
                                rx *= shots[i][2]
                                ry *= shots[i][3]

                                lx += shots[i][0]
                                ly += shots[i][1]
                                rx += shots[i][0]
                                ry += shots[i][1]

                                slice_boxes.append([lx, ly, rx, ry])
                                slice_prob.append(p)

            slice_boxes = np.array(slice_boxes)
            slice_prob = np.array(slice_prob)

            slice_boxes = non_max_suppression(slice_boxes, slice_prob, self.iou_threshold)

            for sb in slice_boxes:
                boxes.append(sb)


        boxes = np.array(boxes)
        boxes = union_suppression(boxes, self.union_threshold)

        return list(boxes)

    def get_boxes(self, image,path=True):
        # Loading the image
        if path:
            image = cv2.imread(image)

        # Just to make sure of proper aspects and colors   
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # let's detect the boundaries
        boxes = self.detect(frame)

        return frame, boxes

    def draw_boundaries(self, image,path=True):
        frame, boxes = self.get_boxes(image,path=path)

        # ploting the image
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111)
        ax.imshow(frame)

        # ploting the boxes, just multiply each predicted [0, 1] relative coordinate to image side in pixels respectively
        for box in boxes:
            x = round(box[0] * frame.shape[1])
            y = round(box[1] * frame.shape[0])
            w = round(box[2] * frame.shape[1])
            h = round(box[3] * frame.shape[0])
            # x, y, w, h here
            ax.add_patch(Rectangle((x,y),w - x,h - y,linewidth=4, edgecolor='blue',facecolor='none'))

    def get_faces(self, image,path=True):
        frame, boxes = self.get_boxes(image,path=path)

        faces = []
        for box in boxes:
            x = round(box[0] * frame.shape[1])
            y = round(box[1] * frame.shape[0])
            w = round(box[2] * frame.shape[1])
            h = round(box[3] * frame.shape[0])
            faces.append((frame[y:h, x:w, :], (x, y, w, h)))

        return frame, faces