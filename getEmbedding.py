import os
import glob
import pickle
from  ArcfaceSshOLnet import FacialRecognition
import cv2
import argparse
import pickle
from matplotlib import pyplot as plt

mtcnn_model= "./model/"
arcface_model= "./model/model-r100-ii/model,0"
ssh_detector="./model/ssh-model-final/sshb"
gpu_index= 1
mtcnn_num_worker=2

model = FacialRecognition(mtcnn_model = mtcnn_model, arcface_model=arcface_model,
                              ssh_detector=ssh_detector, gpu_index=gpu_index, mtcnn_num_worker=2)
flip = False

dicts = []
path_data = "Singer_3_train"
list_people = os.listdir(path_data)
wr = open("singer-nganltp-model-3","wb")
for person in list_people:
#     os.mkdir('detect-face-100/' + str(person))
    print(person)
    path_person = os.path.join(path_data, person)
    list_name_person = os.listdir(path_person)
    error = 0
    for name_img in list_name_person:
        try:
            dict = {}
            path_img = os.path.join(path_person, name_img)
            img = cv2.imread(path_img)
            print(path_img)
            embedding, nimg = model.detect_face_and_get_embedding(img)
            if embedding is None:
                error += 1
            if embedding is not None:
                dict['class'] = person
                dict['features'] = embedding
                dict['imgfile'] = name_img
                dicts.append(dict)
    #         img = cv2.flip(img, 1)
    #         embedding, nimg = model.detect_face_and_get_embedding(img)
    #         if embedding is not None:
    #             dict['class'] = person
    #             dict['features'] = embedding
    #             dict['imgfile'] = name_img
    #             dicts.append(dict)
        except Exception as e:
            print('Caught this error: ' + repr(e))

    print ("So buc anh ko detect duoc face ", error)
pickle.dump(dicts, wr)
wr.close()
print("done")