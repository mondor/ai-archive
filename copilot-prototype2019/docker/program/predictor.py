# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import sys
import signal
import traceback
import torch
from torchvision import datasets, models, transforms
import flask
from flask import request
from flask import jsonify
import pandas as pd
import urllib
import time
import random
from torch.nn import functional as F
from PIL import Image
from io import BytesIO
from torch.autograd import Variable
from torch.autograd import Variable
import re
import string
import traceback
from object_detection import get_object_model, detect_objects
import pretrainedmodels
import pretrainedmodels.utils as utils

model_path = os.path.join('/opt/ml/', 'model')
os.environ['TORCH_MODEL_ZOO'] = model_path

def download_file(url, dest):
    if not os.path.isfile(dest):
        urllib.request.urlretrieve(url, dest)

def get_places_model():
    # download from: http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar
    model_saved_path = os.path.join(model_path, 'resnet18_places365.pth.tar')
    model = models.__dict__['resnet18'](num_classes=365)
    checkpoint = torch.load(model_saved_path, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # download from https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt
    label_saved_path = os.path.join(model_path, 'categories_places365.txt')                    
    labels = list()    
    with open(label_saved_path) as f:
        for line in f:
            parts = line.strip().split(' ')[:-1]
            labels.append(''.join(parts))
            
    labels = tuple(labels)

    return model, labels


def get_inception_model():
    # load model
    model_name = 'inceptionv4'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.eval()
    
        
    # download from https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json 
    label_saved_path = os.path.join(model_path, 'imagenet-simple-labels.json') 
    labels = {}
    with open(label_saved_path) as f:
        data = json.load(f)
        labels = {int(key): value for key, value in enumerate(data)}
         
    return model, labels

    
def read_image(url):
    return urllib.request.urlopen(url).read()
    
    
def to_tensor_places(image_resp):
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(BytesIO(image_resp))
    img = img.convert('RGB')
    img = data_transform(img)
    img = img.unsqueeze(0)
    img = Variable(img, requires_grad=False)
    return img

def to_tensor_inception(image_resp, model):
    img = Image.open(BytesIO(image_resp))
    img = img.convert('RGB')
    tf_img = utils.TransformImage(model) 
    img = tf_img(img)
    img = img.unsqueeze(0)
    img = Variable(img, requires_grad=False)
    return img
    

def validate_image(url):
    available_mime_types = ['image/jpeg', 'image/png']
    
    mime_type = None    
    with urllib.request.urlopen(url) as response:
        info = response.info()
        mime_type = info.get_content_type()
        
    return mime_type in available_mime_types        
        

def predict(model, labels, input_img, threshold=0.5, top_k=3):
    logit = model(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs, idx = probs.numpy(), idx.numpy()
    # output the prediction
    result = {}
    for i in range(0, top_k):
        if i in idx:
            ind = idx[i]
            prob = probs[i]
            label = labels[ind].lower()
            if prob > threshold and label not in result:
                p = round(float(prob), 2)
                result[label] = p

    return result


# global
places_model, places_label = None, None
inception_model, inception_label = None, None
object_model, object_label = None, None

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    return flask.Response(response='\n', status=200, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    try:
        global places_model, places_label, inception_model, inception_label, object_model, object_label

        request_body = request.get_json()
        if "image" not in request_body:
            return jsonify({"error": "image is required"})

        image_resp = read_image(request_body["image"]) 
                   
        if places_model is None:
            t1 = time.time()           
            places_model, places_label = get_places_model()
            t2 = time.time()
            print("Reload Places took {:.2f}s".format(t2 - t1))

        if inception_model is None:
            t1 = time.time()            
            inception_model, inception_label = get_inception_model()
            t2 = time.time()
            print("Reload Inception took {:.2f}s".format(t2 - t1))
            
        if object_model is None:
            t1 = time.time()
            init_net_path = os.path.join(model_path, 'mobilenet-v1-ssd_init_net.pb')
            predict_net_path = os.path.join(model_path, 'mobilenet-v1-ssd_predict_net.pb')
            label_path = os.path.join(model_path, 'voc-model-labels.txt')
            object_model, object_label = get_object_model(init_net_path, predict_net_path, label_path)
            t2 = time.time()
            print("Reload Object Detection Model took {:.2f}s".format(t2 - t1))


        final_result = []
        
        # run Places model
        t1 = time.time()
        result = predict(places_model, places_label, to_tensor_places(image_resp), threshold=0.1, top_k=2)
        t2 = time.time()
        print('Places took {:.2f}s'.format(t2 - t1))
        for label, prob in result.items():
            final_result.append({
                "prob": prob,
                "tag": label,
                "via": "places"
            })

        # run inception model            
        t1 = time.time()
        result = predict(inception_model, inception_label, to_tensor_inception(image_resp, inception_model), threshold=0.1)
        t2 = time.time()
        print('Inception took {:.2f}s'.format(t2 - t1))
        for label, prob in result.items():
            final_result.append({
                "prob": prob,
                "tag": label,
                "via": "inception"
            })

        # run object model   
        t1 = time.time()    
        result = detect_objects(object_model, object_label, image_resp, threshold=0.5)
        t2 = time.time()
        print('Object took {:.2f}s'.format(t2 - t1))       
        for label, prob in result.items():
            final_result.append({
                "prob": prob,
                "tag": label,
                "via": "ssd"
            })


        return jsonify(final_result)
    
    except Exception as e:        
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "image": image_url
        })
    
        
    
   

