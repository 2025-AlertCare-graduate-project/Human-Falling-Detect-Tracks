import cv2
import os

def merge_videos(video_paths, output_path):
    """3개의 영상을 순서대로 병합하여 output_path로 저장"""
    if not video_paths:
        print("[Merge] 병합할 영상이 없습니다.")
        return None

    caps = [cv2.VideoCapture(path) for path in video_paths if os.path.exists(path)]
    if not caps:
        print("[Merge] 열 수 있는 영상이 없습니다.")
        return None

    # 첫 영상 기준으로 파라미터 추출
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for cap in caps:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    print(f"[Merge] 병합 완료: {output_path}")
    return output_path
