import pickle
import numpy as np
import csv
import argparse
from  ArcfaceSshOLnet import FacialRecognition
import os
import cv2
import faiss
from matplotlib import pyplot as plt
from skimage import io

def read_data_train(path):
    f = open(path, "rb")
    dict_ = pickle.load(f)
    X = []
    Y = []
    db_img_paths = []
    for x in dict_:
        _class = (x['class'])
        Y.append(_class)
        X.append(np.array(x['features']))
        db_img_paths.append(x['imgfile'])
    X = np.array(X)
    Y = np.array(Y)
    f.close()
    return X, Y, db_img_paths

mtcnn_model= "./model/"
arcface_model= "./model/model-r100-ii/model,0"
ssh_detector="./model/ssh-model-final/sshb"
gpu_index= 1
mtcnn_num_worker=2
model = FacialRecognition(mtcnn_model = mtcnn_model, arcface_model=arcface_model,
                              ssh_detector=ssh_detector, gpu_index=gpu_index, mtcnn_num_worker=2)
train_embedding_file = 'singer-nganltp-model-3'
threshold = 1.2

X_train, Y_train, db_path = read_data_train(train_embedding_file)
d = 512
search_model = faiss.IndexFlatL2(d)
search_model.add(X_train)

X_train, Y_train, db_path = read_data_train(train_embedding_file)
d = 512
search_model = faiss.IndexFlatL2(d)
search_model.add(X_train)

# TEST 100 
import time
path_data = "DATA-CELEB-100-NGANLTP/test"
list_people = os.listdir(path_data)
accs = []
nones = []
start = time.time()
for person in list_people:
#     os.mkdir('False/' + str(person))
#     dict = {}
    print(person)
    path_person = os.path.join(path_data, person)
    list_name_person = os.listdir(path_person)
    Timg = 0
    Fimg=0
    acc = 0
    none = 0
    for name_img in list_name_person:
        path_img = os.path.join(path_person, name_img)
        print(path_img)
        img = cv2.imread(path_img)
        predicts = set(predict(img))
        if predicts is None:
            none += 1
        if predicts is not None:
            if person in predicts:
                Timg += 1
            else:
                Fimg += 1
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 cv2.imwrite('False/' + str(person) + "/" + str(name_img),img)    
#     acc = Timg/(Timg+Fimg)
    print('Result: ',person,": ",acc, " none: ", none, ' Timg: ', Timg, ' Fimg: ', Fimg)
#     accs.append(acc)
#     nones.append(none)
print(time.time() - start)
# print(accs)
# print(nones)

#         print(path_img)
#         img = cv2.imread(path_img)