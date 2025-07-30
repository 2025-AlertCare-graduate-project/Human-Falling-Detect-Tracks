import time
import torch
import numpy as np
import torchvision.transforms as transforms

from queue import Queue
from threading import Thread

from Detection.Utils import ResizePadding
from ultralytics import YOLO

class YOLO11_onecls(object):
    def __init__(self,
                 input_size=416,
                 weight_file='Models/YOLO/yolo11n.pt',
                 nms=0.2,
                 conf_thres=0.45,
                 device='mps'):
        self.input_size = input_size
        self.model = YOLO(weight_file)
        self.model.eval()
        self.device = device

        self.nms = nms
        self.conf_thres = conf_thres

        self.resize_fn = ResizePadding(input_size, input_size)
        self.transf_fn = transforms.ToTensor()

    def detect(self, image, need_resize=True, expand_bb=5):
        """Feed forward to the model."""
        """
        가장 person 클래스로 confidence가 높은 박스를 찾아서
        원래 Darknet의 non_max_suppression 과정 반환 형식으로 맞춰 줘야 함
        Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
        """
        image_size = (self.input_size, self.input_size)

        # 이미지 리사이즈 (필요 시)
        if need_resize:
            image_size = image.shape[:2]  # 원본 이미지 크기 저장
            image = self.resize_fn(image)  # 이미지 리사이즈

        # 이미지를 텐서로 변환
        image = self.transf_fn(image)[None, ...]
        image = image.to(self.device)
        scf = torch.min(self.input_size / torch.FloatTensor([image_size]), 1)[0]

        # yolo 11 모델 돌리기
        results = self.model(image)

        # 'boxes'에서 바운딩 박스 접근
        boxes = results[0].boxes.xyxy.cpu()  # [left, top, right, bottom]
        confidences = results[0].boxes.conf.cpu()  # 신뢰도
        classes = results[0].boxes.cls.cpu()  # 클래스 정보

        # person (class == 0)만 필터링
        # COCO 데이터셋으로 학습된 모델들은 person 클래스가 0이라 함
        is_person = (classes == 0)
        person_boxes = boxes[is_person]
        person_confidences = confidences[is_person]
        person_classes = classes[is_person]

        # 사람이 detect 되지 않으면 빈 tensor 반환
        if person_boxes.shape[0] == 0:
            print("🔴 사람 클래스가 발견되지 않음")
            return torch.empty((0, 7))

        # 가장 신뢰도 높은 사람 1명 추출
        top_idx = torch.argmax(person_confidences)
        top_person_box = person_boxes[top_idx]
        top_person_conf = person_confidences[top_idx]
        top_person_class = person_classes[top_idx]

        # print("🟢 yolo11n 가상 신뢰도 높은 사람 박스:", top_person_box)

        # 형식 맞추기
        op_person_box_2d = top_person_box.unsqueeze(0)
        top_person_conf_2d = torch.tensor([[top_person_conf]])
        top_person_class_2d = torch.tensor([[top_person_class]])
        detections = torch.cat([op_person_box_2d, top_person_conf_2d, top_person_conf_2d, top_person_class_2d], dim=1)

        # 좌표 복원 및 박스 확장
        detections[:, [0, 2]] -= (self.input_size - image_size[1]) / 2
        detections[:, [1, 3]] -= (self.input_size - image_size[0]) / 2
        detections[:, 0:4] /= scf
        detections[:, 0:2] = np.maximum(0, detections[:, 0:2] - expand_bb)
        detections[:, 2:4] = np.minimum(image_size[::-1], detections[:, 2:4] + expand_bb)

        return detections

class ThreadDetection(object):
    def __init__(self,
                 dataloader,
                 model,
                 queue_size=256):
        self.model = model

        self.dataloader = dataloader
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            images = self.dataloader.getitem()

            outputs = self.model.detect(images)

            if self.Q.full():
                time.sleep(2)
            self.Q.put((images, outputs))

    def getitem(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True

    def __len__(self):
        return self.Q.qsize()







