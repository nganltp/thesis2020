import os
import glob
import pickle
from  ArcfaceRetina import FacialRecognition
import cv2
import argparse
import pickle
from matplotlib import pyplot as plt


arcface_model= "./model/model-r100-ii/model,0"
retina_detector = "./model/R50"
gpu_index= 0

model = FacialRecognition(arcface_model=arcface_model,
                              retina_model=retina_detector, gpu_index=gpu_index)
flip = False


#filter data
from scipy.spatial import distance
import shutil
import numpy as np
path_data = 'Process/Train'
list_people = os.listdir(path_data)
save_as_rm = 'Process/Remove'
save_as_ft = 'Process/Filter'
for person in list_people:
    print(person)
    path_person = os.path.join(path_data, person)
    list_name_person = os.listdir(path_person)
    embedding_avg = 0
    embedding_list = []
    for j, name_img in enumerate(list_name_person):
      try:
        path_img = os.path.join(path_person, name_img)
        print(j, path_img)
        img = cv2.imread(path_img)
        embedding, nimg = model.detect_face_and_get_embedding_test_2(img)
        
        count_face = len(embedding)
        len_em_ls = len(embedding_list)
        dis = distance.euclidean(embedding_avg,embedding[0])

# save as filter
        save_ft = os.path.join(save_as_ft, person)
        try:
            os.mkdir(save_ft)  
        except OSError as error:  
            print(error)  
# save as filter
        save_rm = os.path.join(save_as_rm, person)
        try:
            os.mkdir(save_rm)  
        except OSError as error:  
            print(error)  

        print(count_face, len_em_ls, dis)

        if count_face == 1 and ((len_em_ls == 0) or dis  <= 1.2):
            print('choice')
            embedding_list.append(embedding[0])
            embedding_avg = np.average(embedding_list)
            destination = os.path.join(save_ft, name_img)
            shutil.move(path_img, destination)
        else:
            print('rm')
            save_as = os.path.join(save_as_rm, person)
            destination = os.path.join(save_rm, name_img)
            shutil.move(path_img, destination)
      except:
        print('err', path_img)
