import sys
sys.path
sys.path.append('./pytorch_ssd')

import vision.utils.box_utils_numpy as box_utils
from vision.utils.misc import Timer
from vision.ssd.config.mobilenetv1_ssd_config import specs, center_variance, size_variance
import cv2
from caffe2.python import core, workspace, net_printer
import numpy as np
import time

def get_object_model(init_net_path, predict_net_path, label_path):   
    with open(init_net_path, "rb") as f:
        init_net = f.read()
    with open(predict_net_path, "rb") as f:
        predict_net = f.read()        
    p = workspace.Predictor(init_net, predict_net)    
    class_names = [name.strip() for name in open(label_path).readlines()]        
    return p, class_names

    
def get_boxes(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                  iou_threshold=iou_threshold,
                                  top_k=top_k,
                                  )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
        
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def detect_objects(model, class_names, image_resp, threshold=0.55): 
    orig_image = cv2.imdecode(np.asarray(bytearray(image_resp), dtype="uint8"), cv2.IMREAD_COLOR)
        
    # normalise image    
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))
    image = image.astype(np.float32)
    image = (image - 127) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
   
    confidences, boxes = model.run({'0': image})
    boxes, labels, probs = get_boxes(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    
    result = {}
    for p, l in zip(probs, labels):
        n = class_names[l]
        if n not in result:
            result[n] = round(float(p), 2)
        
    return result
    
    
