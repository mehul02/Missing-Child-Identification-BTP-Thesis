import numpy as np
import pandas as pd
import cv2
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from keras.models import load_model
from PIL import Image

import os
import mtcnn


def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert("RGB")
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    if len(results) == 0:
        return []
    x1, y1, width, height = results[0]["box"]
    # deal with negative pixel index
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


def load_face(dir):
    faces = list()
    # enumerate files
    for filename in os.listdir(dir):
        path = dir + filename
        face = extract_face(path)
        faces.append(face)
    tmp = []
    for i in range(len(faces)):
        if faces[i] != []:
            tmp.append(faces[i])

    return tmp


def load_dataset(dir):
    # list for faces and labels
    X, y = list(), list()
    for subdir in os.listdir(dir):
        path = dir + subdir + "/"
        faces = load_face(path)
        labels = [subdir for i in range(len(faces))]
        print("loaded %d sample for class: %s" % (len(faces), subdir))  # print progress
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


# load train dataset
trainX, trainy = load_dataset("D:/Downloads/BTP/Dataset_BTP/train_final/")
# load test dataset

print(trainX.shape, trainy.shape)

testX, testy = load_dataset("D:/Downloads/BTP/Dataset_BTP/test/")
print(testX.shape, testy.shape)

# save and compress the dataset for further use
np.savez_compressed("D:/Downloads/BTP/face_dataset.npz", trainX, trainy, testX, testy)

##################################################### After saving


data = np.load("D:/Downloads/BTP/face_dataset.npz")
trainX, trainy, testX, testy = (
    data["arr_0"],
    data["arr_1"],
    data["arr_2"],
    data["arr_3"],
)
print("Loaded: ", trainX.shape, trainy.shape, testX.shape, testy.shape)

facenet_model = load_model("D:/Downloads/BTP/Models/model/facenet_keras.h5")
print("Loaded Model")


########################### Embeddings


def get_embedding(model, face):
    # scale pixel values
    face = face.astype("float32")
    # standardization
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(sample)
    return yhat[0]


# convert each face in the train set into embedding
emdTrainX = list()
for face in trainX:
    emd = get_embedding(facenet_model, face)
    emdTrainX.append(emd)

emdTrainX = np.asarray(emdTrainX)
print(emdTrainX.shape)

# convert each face in the test set into embedding
emdTestX = list()
for face in testX:
    emd = get_embedding(facenet_model, face)
    emdTestX.append(emd)
emdTestX = np.asarray(emdTestX)
print(emdTestX.shape)

# save arrays to one file in compressed format
np.savez_compressed(
    "D:/Downloads/BTP/face_embeddings.npz", emdTrainX, trainy, emdTestX, testy
)
############################# Classification on whole dataset

embeddings = np.load("D:/Downloads/BTP/face_embeddings.npz")
emdTrainX, trainy, emdTestX, testy = (
    embeddings["arr_0"],
    embeddings["arr_1"],
    embeddings["arr_2"],
    embeddings["arr_3"],
)

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

print("Dataset: train=%d, test=%d" % (emdTrainX.shape[0], emdTestX.shape[0]))
# normalize input vectors
in_encoder = Normalizer()
emdTrainX_norm = in_encoder.transform(emdTrainX)
emdTestX_norm = in_encoder.transform(emdTestX)
# label encode targets

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy_enc = out_encoder.transform(trainy)
testy_enc = out_encoder.transform(testy)

# fit model
# model = SVC(kernel="linear", probability=True)
# model = xgb.XGBClassifier(objective="binary:logistic", random_state=0)
# model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
model = GaussianNB(priors=None, var_smoothing=1e-9)
model.fit(emdTrainX_norm, trainy_enc)
# predict
yhat_train = model.predict(emdTrainX_norm)
yhat_test = model.predict(emdTestX_norm)
# score
score_train = accuracy_score(trainy_enc, yhat_train)
score_test = accuracy_score(testy_enc, yhat_test)
# summarize
print("Accuracy: train=%.3f, test=%.3f" % (score_train * 100, score_test * 100))

############################### classification

# from random import choice
# select a random face from test set
# selection = choice([i for i in range(testX.shape[0])])


selection = 9
random_face = testX[selection]
random_face_emd = emdTestX_norm[selection]
random_face_class = testy_enc[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

# prediction for the face
samples = np.expand_dims(random_face_emd, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
all_names = out_encoder.inverse_transform([0, 1, 2, 3, 4])

details = {}
details["Ananya"] = {"Address": "123 hazira Gwalior M.P.", "Phone": "8989855234"}
details["Anuj"] = {"Address": "437 Teachers Colony Udaipur Raj.", "Phone": "8989575241"}
details["Gopal"] = {"Address": "232 hazira Gwalior M.P.", "Phone": "7828272472"}
details["Kiran"] = {"Address": "42 Sector 5 Udaipur Raj.", "Phone": "7285828278"}
details["Lovely"] = {"Address": "34 Sector 5 Udaipur Raj.", "Phone": "9898989898"}
details["Mayank"] = {"Address": "4 Sector 5 Udaipur Raj.", "Phone": "9859829454"}
details["Praveen"] = {"Address": "123 hazira Gwalior M.P.", "Phone": "9897821213"}
details["Rakesh"] = {"Address": "23 Sector 8 Udaipur Raj.", "Phone": "9897521232"}
details["Riya"] = {"Address": "333 Sector 1 Vapi Guj.", "Phone": "9232521454"}
details["Saumya"] = {"Address": "545 City Center Gwalior M.P.", "Phone": "9562123354"}
details["Saurabh"] = {"Address": "677 Padav Gwalior M.P.", "Phone": "9874121233"}
details["Suman"] = {"Address": "222 Sector 51 Udaipur Raj..", "Phone": "8979846415"}
details["Suresh"] = {"Address": "2 Sector 52 Kanpur U.p.", "Phone": "9854212154"}

title_x = str(
    str(predict_names[0])
    + " Address: "
    + str(details[str(predict_names[0])]["Address"])
    + " Phone No: "
    + str(details[str(predict_names[0])]["Phone"])
)

"""

"""


print("Predicted: %s (%.3f)" % (predict_names[0], class_probability))
print("Predicted: \n%s \n%s" % (all_names, yhat_prob[0] * 100))
print("Expected: %s" % random_face_name[0])


# plot face
plt.imshow(random_face)
title = "%s (%.3f)" % (predict_names[0], class_probability)
plt.title(title)
plt.show()
