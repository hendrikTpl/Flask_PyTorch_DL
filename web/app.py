# import os
# import requests
# from flask import Flask, request, render_template, redirect
# import pickle
# from werkzeug.utils import secure_filename 

# app = Flask(__name__)

# app.config['IMAGE_UPLOADS'] = os.path.join(os.getcwd(), 'static')

# @app.route('/', methods=['POST', 'GET'])
# def home():
#     return render_template('base.html')

# @app.route('/predict',methods=['POST', 'GET'])
# def predict():
#     """Grabs the input values and uses them to make prediction"""
#     if request.method == 'POST':
#         print(os.getcwd()) 
#         image = request.files["file"]
#         if image.filename == '':
#             print("Filename is invalid")
#             return redirect(request.url)

#         filename = secure_filename(image.filename)

#         basedir = os.path.abspath(os.path.dirname(__file__))
#         img_path = os.path.join(basedir, app.config['IMAGE_UPLOADS'], filename)
#         image.save(img_path)
#         res = requests.post("http://torchserve-mar:8080/predictions/mnist", files={'data': open(img_path, 'rb')})
#         prediction = res.json()
#     return render_template('base.html', prediction_text=f'Predicted Number: {prediction}')

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)

## Library Flask core
from flask import Flask, request, Response, jsonify, render_template
from flask_cors import CORS
from flask_bootstrap import Bootstrap5
# import pickle
from werkzeug.utils import secure_filename
from PIL import Image
import io

## for pytorch model
import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from model.my_model import FCNet, MyCNN


app = Flask(__name__)
CORS(app)


"""
supposedly, you may only need to deploy the ready model, you can also found pre-trained model 
on modelzoo at https://modelzoo.co/ and https://pytorch.org/serve/model_zoo.html
"""

# call the model object from my_model.py

# # Loading the model from the lowest validation loss 
# model_1.load_state_dict(torch.load('FNet_model.pth'))
# model_2.load_state_dict(torch.load('convNet_model.pth'))
 
# Get and Load the pretrained model   
model = MyCNN()
model.load_state_dict(torch.load("./model/mycnn_model.pth"))
print("Loaded model")

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = model.to(device)

# Preprocessing and postprocessing and Normalize the test set same as training set without augmentation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


classes=['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']



"""
API 
"""
# Health check route
@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

@app.route("/")
def home():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    print("/predict request")
    req_json = request.get_json()
    json_instances = req_json["instances"]
    X_list = [np.array(j["image"], dtype="uint8") for j in json_instances]
    X_transformed = torch.cat([transform(x).unsqueeze(dim=0) for x in X_list]).to(device)
    preds = model(X_transformed)
    preds_classes = [classes[i_max] for i_max in preds.argmax(1).tolist()]
    print(preds_classes)
    return jsonify({
        "predictions": preds_classes
    })
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)