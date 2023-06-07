import torch
import cv2
import numpy as np


def get_price(size):
    # 크기에 따른 가격 계산 로직 구현
    # 예시로 일부 크기 범위에 대한 가격을 정의하였습니다.
    if size < 40000:
        price = 100
    elif size < 60000:
        price = 200
    else:
        price = 300

    return price


# YOLOv5 모델 불러오기
model = torch.hub.load("ultralytics/yolov5", "custom", path="path/to/yolov5s.pt")

# 생수병 클래스
class_names = ["bottle"]

# 이미지 읽기
image = cv2.imread("data/image2.jpg")

# 객체 감지
results = model(image)

# 결과 처리
boxes = results.pandas().xyxy[0].values
confidences = results.pandas().xyxy[0]["confidence"].values
labels = results.pandas().xyxy[0]["name"].values

totla_price = 0
for box, confidence, label in zip(boxes, confidences, labels):
    x1, y1, x2, y2 = box[:4].astype(int)
    class_name = label  # 클래스 이름 가져오기
    size = (x2 - x1) * (y2 - y1)
    price = get_price(size)  # 크기에 따라 가격을 가져오는 함수 호출

    totla_price += price

    # 경계 상자 그리기
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 텍스트 추가
    text = f"Price: {price}"
    cv2.putText(
        image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )

text = f"Price: {totla_price}"
cv2.putText(image, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 결과 이미지 저장
cv2.imwrite("result/output.jpg", image)
print("정상완료")
