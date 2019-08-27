from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json

#library of predict
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import cv2
import numpy as np
import urllib
# Create your views here.

@api_view(["POST"])
def predict_disease(link_image):
    try:
        image = json.loads(link_image.body)
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)
    mobile = tf.keras.applications.mobilenet.MobileNet()
    # CREATE THE MODEL ARCHITECTURE

    # Exclude the last 5 layers of the above model.
    # This will include all layers up to and including global_average_pooling2d_1
    x = mobile.layers[-6].output

    # Create a new dense layer for predictions
    # 7 corresponds to the number of classes
    x = Dropout(0.25)(x)
    predictions = Dense(7, activation='softmax')(x)

    # inputs=mobile.input selects the input layer, outputs=predictions refers to the
    # dense layer we created above.

    model = Model(inputs=mobile.input, outputs=predictions)
    model.load_weights("./MyApp/model.h5")
    #print(image["link_image"])
    req = urllib.request.urlopen(image["link_image"])
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr,1)
    img = cv2.resize(img,(224,224))
    img_ex = np.expand_dims(img, axis=0)
    result = model.predict(img_ex)
    result = result.ravel().tolist()
    return JsonResponse(json.dumps(result),safe=False)