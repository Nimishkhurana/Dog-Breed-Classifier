from flask import Flask,request,render_template,redirect
import io
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision.models import vgg16,densenet121
from torchvision import transforms
import PIL
from PIL import Image
from torch.autograd import Variable

app = Flask(__name__)


import pickle
with open('class_names.pickle', 'rb') as handle:
    class_names = pickle.load(handle)

torch.set_default_tensor_type(torch.FloatTensor)

def image_loader(image_bytes,loader):
    """load image, returns cuda tensor"""
    image = Image.open(io.BytesIO(image_bytes))
    image = loader(image).float()
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image.to('cpu')


data_transforms = transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_face(image_bytes):
    img_stream = io.BytesIO(image_bytes)
    img = cv2.imdecode(np.fromstring(img_stream.read(), np.uint8), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def detect_dog(img):
  img = image_loader(img,data_transforms)
  with torch.no_grad():
      output = model2.forward(img)
  top_p,top_class = output.topk(1,dim=1)
  if(top_class>=151 and top_class <=268):
      return True
  else:
      return False

num_classes = 133


model1 = vgg16(pretrained = False)
model1.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)
state_dict = torch.load('checkpoint_dogbreed.pth',map_location=lambda storage, location: storage)
model1.load_state_dict(state_dict,strict=False)
model1.eval()
model1.to('cpu')

model2 = densenet121(pretrained=True)
model2.eval()
model2.to('cpu')

model3 = densenet121(pretrained=True)
model3.classifier = nn.Linear(in_features=1024, out_features=num_classes)
state_dict2 = torch.load('checkpoint_densenet121_classifier.pth',map_location=lambda storage, location: storage)
model3.classifier.load_state_dict(state_dict2)
model3.to('cpu')
model3.eval()


def predict(image_bytes,model):
    image = image_loader(image_bytes,data_transforms)
    with torch.no_grad():
        output = model(image)
    top_p,top_class = output.topk(1,dim=1)
    print(top_class)
    pred_class = class_names[top_class]
    print(pred_class)

    return pred_class

def format_output(pred):

    pred = pred.replace('-',' ')
    pred = pred.replace('_',' ')
    return pred


@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        image_recieved=file.read()
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if detect_dog(image_recieved):
            pred_class_vgg=predict(image_recieved,model1)[4:]
            pred_class_densenet=predict(image_recieved,model3)[4:]
            pred_class = pred_class_vgg + " (VGG16) and " + pred_class_densenet +" (Densenet121)"
        elif detect_face(image_recieved):
            similar_class_vgg = predict(image_recieved,model1)[4:]
            similar_class_densenet = predict(image_recieved,model3)[4:]
            pred_class = "Hello human.You look like" + similar_class_vgg +" (VGG16) and " + similar_class_densenet+ " (Densenet121)"
        else:
            pred_class="Not Dog.Trying to fool me hmm!!"

        pred_class = format_output(pred_class)
        return render_template('index.html',name=pred_class)
        # return render_template('after.html',n1=names[0].capitalize(),n2=names[1].capitalize(),p1=prob[0],p2=prob[1])
    return render_template('index.html')
