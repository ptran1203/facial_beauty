

from keras.models import load_model, model_from_json
import urllib.request
import cv2
from PIL import Image
import numpy as np
import time


def get_model():
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model/model_weight.h5")
    return model

now = time.time()
MODEL = get_model()
print('get model take: ', time.time() - now)
def predict(img_path):
    img = None
    # get image from internet
    if img_path.startswith('http') or \
       img_path.startswith('data:image/jpeg'):
        req = urllib.request.urlopen(img_path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
    # get image from local file
    else:
        img = cv2.imread(img_path)

    # resize and rescale to fit model 
    img = cv2.resize(img, (227, 227)) / 255.0
    img = np.expand_dims(img, axis=0)
    return MODEL.predict(img)[0]

img_path = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSMxRw96-QdObv0SflMIXVmLhFT-QgX4SPURqU4qY_h4FK6w2nL8w&s'
now = time.time()
pre = predict(img_path)
print('predict take: ', time.time() - now)
print(pre)
# print(getsizeof(MODEL))