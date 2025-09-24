import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Modules.Video.CameraLoader import CamLoader_Q
from Modules.Visualize.fn import draw_single
from Modules.Detect.DetectorLoader_yolo11 import YOLO11_onecls
from Modules.Detect.DetectorLoader import TinyYOLOv3_onecls
from Modules.Pose.PoseEstimateLoader import SPPE_FastPose
from Track.Tracker import Detection, Tracker
from Modules.Action.ActionsEstLoader import TSSTG
from main import kpt2bbox
from Detection.Utils import ResizePadding


def load_ground_truth(annotation_path, n_frames):
    """
    Load frame-level labels from Le2i annotation files.
    Fall frames=1, Not-Fall=0
    If annotation_path is None or does not exist, return all zeros.
    """
    labels = [0] * n_frames
    if annotation_path is None or not os.path.exists(annotation_path):
        return labels

    with open(annotation_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 1 and parts[0].lower() == "fall":
            labels = [1] * n_frames
        elif len(parts) == 2:  # start_frame end_frame
            start, end = map(int, parts)
            for i in range(start, min(end + 1, n_frames)):
                labels[i] = 1
    return labels

def run_pipeline(video_path, detect_model, pose_model, tracker, action_model, inp_dets=384, show_frames=False):
    resize_fn = ResizePadding(inp_dets, inp_dets)

    def preproc(image):
        image = resize_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    cam = CamLoader_Q(video_path, queue_size=1000, preprocess=preproc).start()
    prob_vectors = []

    fps_time = 0
    f = 0

    while cam.grabbed():
        f += 1
        frame = cam.getitem()

        # Detection
        detected = detect_model.detect(frame, need_resize=True, expand_bb=10)

        # 빈 detection 체크
        if detected is None or detected.shape[0] == 0:
            print("사람이 감지되지 않았습니다.")
            continue

        # Tracker 예측
        tracker.predict()
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []
        if detected is not None:
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]
        for bb in detected[:, 0:5]:
            frame = cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 1)
        tracker.update(detections)

        # Action Recognition
        for track in tracker.tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)

                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)

            # Visualize
            if show_frames and track.is_confirmed() and len(track.keypoints_list) > 0:
                frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, 'Frame #%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = cv2.putText(frame, 'File %s : ' % video_path,
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = cv2.putText(frame, 'Model %s : ' % detect_model,
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frame = frame[:, :, ::-1]
        fps_time = time.time()

        if show_frames:
            cv2.imshow(f"Fall Detectetion", frame)
            key = cv2.waitKey(1)  # 0이면 키 입력 대기, 이외의 수는 자동으로 진행 (숫자가 작을수록 빠름)
            if key & 0xFF == ord('q'):
                cam.stop()
                cv2.destroyAllWindows()
                return np.array(prob_vectors)

        f += 1

    cam.stop()
    cv2.destroyAllWindows()
    return np.array(prob_vectors)

def evaluate_video(video_path, annotation_path, detect_model, pose_model, tracker, action_model):
    resize_fn = ResizePadding(inp_dets, inp_dets)

    def preproc(image):
        image = resize_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    cam = CamLoader_Q(video_path, queue_size=1000, preprocess=preproc).start()

    n_frames = int(cam.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    cam.stop()

    gt_labels = load_ground_truth(annotation_path, n_frames)
    prob_vectors = run_pipeline(video_path, detect_model, pose_model, tracker, action_model, show_frames=True)

    if len(prob_vectors) == 0:
        return None

    pred_probs = prob_vectors[:, 6]
    preds = (pred_probs > 0.5).astype(int)
    min_len = min(len(gt_labels), len(preds))
    y_true = gt_labels[:min_len]
    y_pred = preds[:min_len]

    metrics = {
        "video": video_path,
        "frames": min_len,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select YOLO model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo11", "tinyyolo"],
        default="tinyyolo",
        help="Choose which model to use: 'yolo11' or 'tinyyolo'"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps"

    # 모델 초기화
    inp_dets = 384

    if args.model == "yolo11":
        detect_model = YOLO11_onecls(inp_dets, device=device)
        print("YOLO 11 모델")
    else:
        detect_model = TinyYOLOv3_onecls(inp_dets, device=device)
        print("Tiny YOLO 모델")

    pose_model = SPPE_FastPose("resnet50", 224, 160, device=device)
    tracker = Tracker(max_age=99999, n_init=1)
    action_model = TSSTG()

    dataset_root = "../Dataset/Le2i/archive"
    results = []

    # archive 하위 모든 폴더 순회
    start_time = time.time()
    print("시간 측정 시작")

    for scenario in os.listdir(dataset_root):
        scenario_path = os.path.join(dataset_root, scenario)
        inner_folder = os.path.join(scenario_path, scenario)
        if not os.path.exists(inner_folder):
            continue

        video_dir = os.path.join(inner_folder, "Videos")
        anno_dir = os.path.join(inner_folder, "Annotation_files")
        if not os.path.exists(video_dir):
            video_dir = inner_folder
            anno_dir = None


        for fname in os.listdir(video_dir):
            if not fname.endswith(".mp4") and not fname.endswith(".avi"):
                continue

            video_path = os.path.join(video_dir, fname)
            annotation_path = None
            if anno_dir and os.path.exists(anno_dir):
                annotation_path = os.path.join(anno_dir, os.path.splitext(fname)[0] + ".txt")

            print(f"🟧 Evaluating {video_path} ...")

            tracker = Tracker(max_age=99999, n_init=1) # tracker 초기화

            metrics = evaluate_video(video_path, annotation_path,
                                     detect_model, pose_model, tracker, action_model)
            if metrics:
                results.append(metrics)

    end_time = time.time()
    print(f"모델 실행 시간: {end_time - start_time:.4f}초")
