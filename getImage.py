import os
import glob
import pickle
from  ArcfaceRetina import FacialRecognition
import cv2
import argparse
import pickle
from matplotlib import pyplot as plt



path_data = 'Process/Filter'
list_people = os.listdir(path_data)
save_as_ft = 'Process/Image'
for person in list_people:
    print(person)
    path_person = os.path.join(path_data, person)
    list_name_person = os.listdir(path_person)
    for j, name_img in enumerate(list_name_person):
        path_img = os.path.join(path_person, name_img)
        print(j, path_img)
        save_ft = os.path.join(save_as_ft, person)
        try:
            os.mkdir(save_ft)  
        except OSError as error:  
            print(error)  
        destination = os.path.join(save_ft, name_img)
        shutil.move(path_img, destination)
        break

          
