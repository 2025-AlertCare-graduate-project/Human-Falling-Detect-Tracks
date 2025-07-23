import torch
import cv2
from ultralytics import YOLO


class UltralyticsPoseFromBBoxes:
    def __init__(self,
                 model_path='yolo11n-pose.pt',
                 input_size=(256, 320),
                 device='mps'):

        self.device = device
        self.model = YOLO(model_path).to(device)
        self.input_size = input_size
        self.model.eval()

    def predict(self, image, bboxs, bboxs_scores):
        """
        bboxs: (1, 4) 크기의 배열 [x1, y1, x2, y2]
        bboxs_scores: (1,) 배열
        """
        if len(bboxs) == 0:
            return []

        # 첫 번째 사람만 사용
        box = bboxs[0]
        x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2]

        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            return []

        # Resize crop to model's input size
        cropped_resized = cv2.resize(cropped, self.input_size)
        result = self.model(cropped_resized, verbose=False)[0]

        if result.keypoints is None or len(result.keypoints) == 0:
            return []

        kpts = result.keypoints.xy[0].cpu().numpy()  # (17, 2)
        scores = result.keypoints.conf[0].cpu().numpy()  # (17,)
        score_mean = scores.mean()

        # 원래 좌표계로 복원
        scale_x = (x2 - x1) / self.input_size[0]
        scale_y = (y2 - y1) / self.input_size[1]
        kpts[:, 0] = kpts[:, 0] * scale_x + x1
        kpts[:, 1] = kpts[:, 1] * scale_y + y1

        # print("🔴")
        # print("keypoints = ", torch.tensor(kpts))
        # print("kp_score = ", torch.tensor(scores))
        # print("score = ", torch.tensor(score_mean))

        return [{
            'keypoints': torch.tensor(kpts), # 2차원
            'kp_score': torch.tensor(scores).unsqueeze(1), # 2차원
            'score': torch.tensor(score_mean)
        }]