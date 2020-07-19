from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import tensorflow as tf
import keras as k
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.titlesize'] = 8
import h5py
import glob
#import cv2
import os
from datetime import date

model = load_model('fruit_classify_model.h5')
size = 512

def predictFruitClass(ImagePath, trainedModel, class_dict):
    """
    Perform class prediction on input image and print predicted class.

    Args:
        ImagePath(str): Absolute Path to test image
        trainedModel(object): trained model from method getTrainedModel()
        DictOfClasses(dict): python dict of all image classes.

    Returns:
        Probability of predictions for each class.
    """
    x = image.load_img(ImagePath, target_size=(size,size))
    x = image.img_to_array(x)
    # for Display Only
    # import matplotlib.pyplot as plt
    # plt.imshow((x * 255).astype(np.uint8))
    x = np.expand_dims(x, axis=0)
    prediction_class = trainedModel.predict_classes(x, batch_size=1)
    prediction_probs = trainedModel.predict_proba(x, batch_size=1)
    print('probs:', prediction_probs)
    print('class_index:', prediction_class[0])
    for key, value in class_dict.items():
        if value == prediction_class.item():
            # return {'class': key, 'confidence': str(prediction_probs[0][prediction_class[0]])}
            return key
    return None

PATH_TO_TRAINED_MODEL_FILE = 'fruit_classify_model.h5'
trained_model_path = PATH_TO_TRAINED_MODEL_FILE
trained_model = load_model('fruit_classify_model.h5')
class_dict = np.load('class_dict.npy', allow_pickle=True).item()

# image_path = './test/rottenb1.jpg'
# single_pred = predictFruitClass(image_path,trained_model, class_dict)
# print(single_pred)

