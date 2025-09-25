import boto3
import requests
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION     = os.getenv('AWS_REGION')
BUCKET_NAME    = os.getenv('BUCKET_NAME')

s3 = boto3.client(
    's3',
    aws_access_key_id     = AWS_ACCESS_KEY,
    aws_secret_access_key = AWS_SECRET_KEY,
    region_name           = AWS_REGION
)

def upload_video(local_path, s3_folder='videos'):
    filename = os.path.basename(local_path)
    key = f"{s3_folder}/{uuid.uuid4().hex}_{filename}"
    s3.upload_file(local_path, BUCKET_NAME, key)
    url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return url

<<<<<<< HEAD:Modules/Video/s3_utils.py
# SPRING_URL = 'http://localhost:8080/api/v1/videos'  # 로컬
SPRING_URL = 'http://3.34.137.110:8080/api/v1/videos' # IP
=======
SPRING_URL = 'http://3.34.137.110:8080/api/v1/videos'  #엔드포인트

>>>>>>> minseo:s3_utils.py

def send_url(video_url, phone_num, fall_detected, detected_time):
    payload = {
        'videoUrl': video_url,
        'careReceiverPhoneNumber' : phone_num,
        'fallDetected': fall_detected,  # 또는 'fall_detected'로도 가능 (서버 쪽 JSON 필드 이름에 맞춰야 함)
        'detectedTime' : detected_time
    }
    resp = requests.post(SPRING_URL, json=payload)
    if resp.status_code == 200:
        print(f"[Spring] 저장 성공: {video_url}")
    else:
        print(f"[Spring] 저장 실패 ({resp.status_code}): {resp.text}")