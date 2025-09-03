import requests
from datetime import datetime

def send_summary(hour_summary, phone_num):
    hour_payload = {
        'phoneNumber' : phone_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "activeTime": hour_summary.get("active", 0),
        "sittingTime": hour_summary.get("sitting", 0),
        "lyingTime": hour_summary.get("lying", 0),
    }

    try:
        requests.post("http://3.34.137.110:8080/api/v1/activity/hourly", json=hour_payload)
    except Exception as e:
        print("[ERROR] 1시간 데이터 전송 실패:", e)

