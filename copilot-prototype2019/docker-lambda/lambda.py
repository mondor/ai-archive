import os
import shutil
import sys
import zipfile
import boto3

artifacts_bucket = 'copilot-lite'

s3 = boto3.client('s3')

# download dependencies: pytorch, numpy etc...
dependencies_zip = "/tmp/pytorch1.0.1_lambda_deps.zip"
dependencies_dir = '/tmp/dependencies'
dependencies_tmp = '/tmp/_dependencies'

sys.path.append(dependencies_dir)

if not os.path.exists(dependencies_dir):
    if os.path.exists(dependencies_tmp):
        shutil.rmtree(dependencies_tmp)    
        
    s3.download_file(artifacts_bucket, "v1/pytorch1.0.1_lambda_deps.zip", dependencies_zip)
    zipfile.ZipFile(dependencies_zip, 'r').extractall(dependencies_tmp)
    os.remove(dependencies_zip)
    os.rename(dependencies_tmp, dependencies_dir)
    print("Dependencies extracted successfully!")

    
import json
import torch
from torchvision import datasets, models, transforms
import urllib
import random
from torch.nn import functional as F
from PIL import Image
from torch.autograd import Variable
from torch.autograd import Variable
import re
import string
import time

model_path = '/tmp/model'

os.environ['TORCH_MODEL_ZOO'] = model_path
if not os.path.exists(model_path):
    os.makedirs(model_path)
    
def download_file(url, dest):
    if not os.path.isfile(dest):
        urllib.request.urlretrieve(url, dest)
        

def get_places_model():
    arch = 'resnet18'

    # load the pre-trained weights
    url = 'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar'
    model_saved_path = os.path.join(model_path, os.path.basename(url))

    download_file(url, model_saved_path)    
    
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_saved_path, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # download labels
    url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    label_saved_path = os.path.join(model_path, os.path.basename(url))
    download_file(url, label_saved_path)    
                
    labels = list()    
    with open(label_saved_path) as f:
        for line in f:
            labels.append(line.strip().split(' ')[0][3:])            
            
    labels = tuple(labels)

    return model, labels


def get_basic_model():
    model = models.inception_v3(pretrained=True)
    model.eval()
    
    url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    label_saved_path = os.path.join(model_path, os.path.basename(url))    
    download_file(url, label_saved_path)
    
    labels = {}
    with open(label_saved_path) as f:
        data = json.load(f)
        labels = {int(key): value for key, value in enumerate(data)}
         
    return model, labels
    

def get_image_tensor(url):    
    image_file = ''.join([random.choice('abcde') for _ in range(6)]) + str(int(time.time()))
    image_saved_path = os.path.join('/tmp', image_file)
    
    urllib.request.urlretrieve(url, image_saved_path)
    
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
#         transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_saved_path)    
    img = img.convert('RGB')    
    img = data_transform(img)    
    img = img.unsqueeze(0)
    img = Variable(img)

    return img


def validate_image(url):
    available_mime_types = ['image/jpeg', 'image/png']
    
    mime_type = None    
    with urllib.request.urlopen(url) as response:
        info = response.info()
        mime_type = info.get_content_type()
        
    return mime_type in available_mime_types        
        

def predict(model, labels, input_img, threshold=0.5):
    logit = model(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs, idx = probs.numpy(), idx.numpy()
    # output the prediction
    result = []
    for i in range(0, 10):
        ind = idx[i]       
        if probs[i] > threshold:
            p = round(float(probs[i]), 2)
            result.append((p, re.sub(r'[^a-zA-Z0-9]', ' ', labels[ind])))
            
    return result


place_model, place_label = None, None
basic_model, basic_label = None, None
   
def lambda_handler(event, context):
    global place_model, place_label, basic_model, basic_label
    
    if "image" not in event:
        return {
            'statusCode': 400,
            'body': json.dumps({
                "error": "image not exists"
            })
        }
    
    if place_model is None:
        place_model, place_label = get_places_model()
        print("place model loaded")
    
    if basic_model is None:    
        basic_model, basic_label = get_basic_model()
        print("basic model loaded")
    
    image_tensor = get_image_tensor(event['image'])    
    
    final_result = []
    result = predict(place_model, place_label, image_tensor, threshold=0.05)
    for prob, label in result:
        final_result.append({
            "prob": prob,
            "tag": label,
            "via": "places365"
        })
    
    result = predict(basic_model, basic_label, image_tensor, threshold=0.3)
    for prob, label in result:
        final_result.append({
            "prob": prob,
            "tag": label,
            "via": "inception_v3"
        })

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps(final_result)
    }
