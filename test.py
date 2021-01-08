import pickle
import numpy as np
import csv
import argparse
from  ArcfaceRetina import FacialRecognition
import os
import cv2
import faiss
from matplotlib import pyplot as plt
from skimage import io
from multiprocessing.dummy import Pool as ThreadPool


def handle_recognize_face(bounded_face, k):
    D, I = search_model.search(bounded_face, k)

    predictions = []
    for k in range(len(I[0])):
        la = Y_train[I[0][k]]
        dis = D[0][k]
        if dis > threshold and 'unknown' not in predictions:
            predictions.append('unknown')
        predictions.append(la)
    return None if not len(predictions) else predictions[0]


def predict_2(img):
    threshold = 1.2
    k = 15
    features, bboxes = model.detect_face_and_get_embedding_test_2(img)
    list_label = []
    if not features:
        return None

    param_pools = []
    for feature in features:
        bounded_face = np.array([feature])

        param_pools.append((bounded_face, k))
    
    pool = ThreadPool()
    pool_results = pool.starmap(
        handle_recognize_face, param_pools
    )
    pool.close()
    pool.join()

    for result in pool_results:
        if result:
            list_label.append(result)

    print("list label: ",list_label)
    return list_label


def read_data_train(path):
    f = open(path, "rb")
    dict_ = pickle.load(f)
    X = []
    Y = []
    db_img_paths = []
    # print(dict_)
    for x in dict_:
        _class = (x['class'])
        Y.append(_class)
        X.append(np.array(x['features']))
        db_img_paths.append(x['imgfile'])
    X = np.array(X)
    Y = np.array(Y)
    f.close()
    return X, Y, db_img_paths


arcface_model= "./model/model-r100-ii/model,0"
retina_detector="./model/R50"
gpu_index= 0
model = FacialRecognition(arcface_model=arcface_model,
                              retina_model=retina_detector, gpu_index=gpu_index)
train_embedding_file = 'Model-v1'
threshold = 1.2

X_train, Y_train, db_path = read_data_train(train_embedding_file)
d = 512
search_model = faiss.IndexFlatL2(d)
search_model.add(X_train)

path_img1 = 'Process/000474_0.jpg'

path_img = 'Process/000196_5.jpg'
# gc.collect()
img = cv2.imread(path_img)
_labels = predict_2(img)
# print(labels)
# gc.collect()

img1 = cv2.imread(path_img1)
labels1 = predict_2(img1)
# print(labels1)