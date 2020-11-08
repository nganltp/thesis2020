import os
import glob
import pickle
from  ArcfaceSshOLnet import FacialRecognition
import cv2
import argparse
import pickle
from matplotlib import pyplot as plt
from shutil import copyfile



arcface_model= "./model/model-r100-ii/model,0"
retina_detector="./model/R50"
gpu_index= 0

model = FacialRecognition(arcface_model=arcface_model,
                              retina_model=retina_detector, gpu_index=gpu_index)
flip = False

dicts = []
path_data = './Data_Celeb/train'
list_people = os.listdir(path_data)
wr = open("Celeb-model-1","wb")

path_else = './Data_Celeb/else'
count = 0
for person in list_people:
#     os.mkdir('detect-face-100/' + str(person))
    print(person)
    path_person = os.path.join(path_data, person)
    list_name_person = os.listdir(path_person)
    error = 0

    folder_else = os.path.join(path_else, person)
    
    for name_img in list_name_person:
        try:
            dict = {}
            path_img = os.path.join(path_person, name_img)
            img = cv2.imread(path_img)
            print(path_img, count)
            embedding, nimg = model.detect_face_and_get_embedding(img)
            count += 1
            # print(embedding, nimg)
            if embedding is None:
                error += 1
            if embedding is not None:
                #print(len(embedding.shape))
                if len(embedding) == 1:
                  dict['class'] = person
                  dict['features'] = embedding
                  dict['imgfile'] = name_img
                  dicts.append(dict)
                if len(embedding.shape) != 1:
                    if not os.path.exists(folder_else):
                      os.makedirs(folder_else)
                    else:    
                      print("Directory already exists")
                    print(os.path.join(path_else, person, name_img))
                    copyfile(path_img, os.path.join(path_else, person, name_img))

    #        
        except Exception as e:
            print('Caught this error: ' + repr(e))

    print ("So buc anh ko detect duoc face ", error)
pickle.dump(dicts, wr)
wr.close()
print("done")