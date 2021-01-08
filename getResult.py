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

img_txt = open("Img_v2.txt","w+") 
person_txt = open("Result_v2.txt","w+") 

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

def predict(img):
    features, bboxes = model.detect_face_and_get_embedding_test_2(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #   GET PREDICTIONS
    k = 15
    predicts = []
    bboxes_results = []
    if features is None:
        return None
    if features is not None:
        for i, feature in enumerate(features):
            D, I = search_model.sreach(np.array([feature]), k)
            predictions = []
            la = Y_train[I[0][0]]
            dis = D[0][0]
            if dis > threshold:
                predictions.append('unknow')
            predictions.append(la)
            if len(predictions) != 0:
                predicts.append(predictions[0])
                bboxes_results.append(bboxes[i])
        return predicts


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

# TEST
count = 1
import time
path_data = u"Process/Test"
list_people = os.listdir(path_data)
accs = []
nones = []
start = time.time()
for person in list_people:
    print(person)
    path_person = os.path.join(path_data, person)
    list_name_person = os.listdir(path_person)
    T_img = 0
    F_img = 0
    acc = 0
    none = 0
    for name_img in list_name_person:
        path_img = os.path.join(path_person, name_img)
        print(path_img, count)
        count += 1
        img = cv2.imread(path_img)
        
        labels = predict_2(img)
        if labels is None:
            none += 1
            img_txt.write(str(path_img) + " none" + "\n")
        if labels is not None:
            for la in labels:
                img_txt.write(str(path_img) + ": "+ str(la) + "\n")
                #person = person.encode("ascii", "ignore").decode()
                # person = unidecode(person)
                #print(type(person) ,str(person), len(str(person)))
                # la = unidecode(la)
                #print(type(la), str(la), len(str(la)))
                #for i, c in enumerate(person):
                #    print(i,c)
                #for i, c in enumerate(la):
                #    print(i,c)
               # print('person: ' + person, 'predict: ' + la)
                if la == person:
                    T_img += 1
                    print('True') 
                else:
                    F_img += 1
                    print('False') 
    total = T_img + F_img
    if (total == 0):
        acc = 0
    else:
        acc = T_img/(T_img + F_img)
    print('Result: ',person," Acc: ",acc, " none: ", none, ' True img: ', T_img, ' False img: ', F_img)
    person_txt.write("Result: " + str(person) + " Acc: " + str(acc) + " none: " + str(none) + ' True img: ' + str(T_img) + ' False img: ' + str(F_img) + "\n")
#     accs.append(acc)
#     nones.append(none)
print(time.time() - start)

img_txt.close()
person_txt.close()
# print(accs)
# print(nones)
