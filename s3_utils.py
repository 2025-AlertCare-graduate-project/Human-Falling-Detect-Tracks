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

def upload_video_to_s3(local_path, s3_folder='videos'):
    filename = os.path.basename(local_path)
    key = f"{s3_folder}/{uuid.uuid4().hex}_{filename}"
    s3.upload_file(local_path, BUCKET_NAME, key)
    url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return url

