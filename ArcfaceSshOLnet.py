# coding=utf-8
import mxnet as mx
import numpy as np
from Arcface import ArcfaceModel
from face_preprocess import preprocess
import cv2
from Sshdetector import SSHDetector
from OnetLnet import OnetLnetAlignment


class FacialRecognition():
    def __init__(self, gpu_index=-1, mtcnn_model="mtcnn-model", arcface_model="model-r100-ii/model,0",
                 image_size='112,112', ssh_detector="ssh-model-final/sshb", mtcnn_num_worker=1):
        if gpu_index >= 0:
            mtcnn_ctx = mx.gpu(gpu_index)
        else:
            mtcnn_ctx = mx.cpu()
        self.face_detector = SSHDetector(prefix=ssh_detector, epoch=0, ctx_id=gpu_index, test_mode=True)
        self.face_recognition = ArcfaceModel(gpu=gpu_index, model=arcface_model, image_size=image_size)
        self.landmark_detector = OnetLnetAlignment(model_folder=mtcnn_model, ctx=mtcnn_ctx, num_worker=mtcnn_num_worker,
                                                   accurate_landmark=True, threshold=[0.6, 0.7, 0.5])

    def get_scales(self, img):
        TEST_SCALES = [100, 200, 300, 400]
        target_size = 400
        max_size = 1200
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        scales = [float(scale) / target_size * im_scale for scale in TEST_SCALES]
        return scales

    def detect_face_and_get_embedding(self, img):
        thresh = 0.2
        scales = self.get_scales(img)
        bboxes = self.face_detector.detect(img, threshold=thresh, scales=scales)
#         print('bbox:', bboxes)
        if len(bboxes) <= 0:
            return None, None
        rs = self.landmark_detector.detect_landmark(img, bboxes)
        if rs is not None:
            _, points = rs
            point = points[0, :].reshape((2, 5)).T
            nimg = preprocess(img, bboxes[0], point, image_size='112,112')
#             nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)      
            x = np.transpose(nimg, (2, 0, 1))
            embeddings = self.face_recognition.get_feature(x)
            return embeddings, nimg
        return None, None
    
    def detect_face_and_get_embedding_test(self, img):
        thresh = 0.2
        scales = self.get_scales(img)
        bboxes = self.face_detector.detect(img, threshold=thresh, scales=scales)
        if len(bboxes) <= 0:
            return None
        print('len bboxes: ',len(bboxes))
        rs = self.landmark_detector.detect_landmark(img, bboxes)
        embeddings = []
        if rs is not None:
            bboxes , points = rs
#             print('len total bboxes: ',len(bboxes))
            for i, bbox in enumerate(bboxes):
#                 print('bbox: ', bbox)
                point = points[i, :].reshape((2, 5)).T
#                 print('point: ', point)
                nimg = preprocess(img, bbox, point, image_size='112,112')
#                 nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                x = np.transpose(nimg, (2, 0, 1))
                embedding = self.face_recognition.get_feature(x)
                embeddings.append(embedding)
                cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]), (255,0,0), 2)
            return embeddings
        return None
    def detect_face_and_get_embedding_test_2(self, img):
        thresh = 0.2
        scales = self.get_scales(img)
        bboxes = self.face_detector.detect(img, threshold=thresh, scales=scales)
        if len(bboxes) <= 0:
            return None, None
        rs = self.landmark_detector.detect_landmark(img, bboxes)
        embeddings = []
        bbox_list = []
        if rs is not None:
            bboxes, points = rs
            for i, bbox in enumerate(bboxes):
                point = points[i, :].reshape((2, 5)).T
                nimg = preprocess(img, bbox, point, image_size='112,112')
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                x = np.transpose(nimg, (2, 0, 1))
                embedding = self.face_recognition.get_feature(x)
                embeddings.append(embedding)
                bbox_list.append(bbox)
            return embeddings, bbox_list
        return None, None

    def get_embedding(self, img):
        nimg = cv2.resize(img, (112, 112))
#         nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        x = np.transpose(nimg, (2, 0, 1))
        embeddings = self.face_recognition.get_feature(x)
        return embeddings
