"""
과제 01: SORT 알고리즘을 활용한 다중 객체 추적기 구현

YOLOv3 객체 검출기와 SORT 추적기를 결합하여
비디오에서 다중 객체를 실시간으로 추적하고 시각화합니다.

핵심 구성:
  - 객체 검출: OpenCV DNN 모듈 + YOLOv3 사전 훈련 모델
  - 객체 추적: SORT (칼만 필터를 이용한 상태 예측 + 헝가리안 알고리즘을 이용한 데이터 연관)
  - 시각화: 각 객체에 고유 ID, 클래스 이름, 경계 상자, 이동 궤적 표시

SORT 동작 원리 (강의 내용 기반):
  1. 검출(Detection): YOLOv3로 현재 프레임의 바운딩 박스를 찾는다.
  2. 예측(Prediction): 칼만 필터가 이전 프레임의 속도 정보를 바탕으로 현재 위치를 예측한다.
  3. 거리 계산: 예측 박스와 검출 박스 간의 IoU를 구해 비용 행렬(Cost Matrix)을 만든다.
  4. 매칭(Matching): 헝가리안 알고리즘으로 전체 비용이 최소가 되는 최적의 박스 쌍을 맺는다.
  5. 관리(Management): 매칭된 트랙은 업데이트, 미매칭 검출은 새 트랙 생성, 오래된 트랙은 삭제.

SORT의 한계:
  - 재식별(Re-ID) 신경망이 없어서, 가림(Occlusion) 발생 후 다시 나타나면
    동일 객체를 인식하지 못하고 새로운 ID를 부여한다 (ID Switch 발생).
  - 이는 코딩 에러가 아닌 알고리즘적 한계이다.
"""

import cv2 as cv
import numpy as np
import os
import time
from collections import defaultdict, deque

# 같은 디렉터리의 sort.py 모듈에서 Sort 클래스를 가져온다
from sort import Sort

# ============================================================
# 경로 설정
# ============================================================
# 현재 스크립트가 위치한 디렉터리를 기준으로 모든 경로를 설정한다
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_files")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# 입력 비디오 경로 (교수님 제공 파일)
VIDEO_PATH = os.path.join(INPUT_DIR, "slow_traffic_small.mp4")

# YOLOv3 모델 파일 경로
YOLO_WEIGHTS = os.path.join(INPUT_DIR, "yolov3.weights")
YOLO_CFG = os.path.join(INPUT_DIR, "yolov3.cfg")
COCO_NAMES = os.path.join(BASE_DIR, "coco.names")

# 결과 저장 경로
OUTPUT_VIDEO = os.path.join(RESULT_DIR, "01_sort_tracking.mp4")

# ============================================================
# YOLOv3 설정 상수
# ============================================================
CONF_THRESHOLD = 0.5   # 검출 신뢰도 임계값 (이 값 이상만 검출로 인정)
NMS_THRESHOLD = 0.4    # Non-Maximum Suppression 임계값 (중복 박스 제거 기준)
INPUT_SIZE = 416       # YOLOv3 입력 해상도 (416x416 고정)

# 궤적(trajectory) 그리기 설정
# 각 트랙의 최근 N개 위치를 저장하여 이동 궤적을 선으로 그린다
TRAJECTORY_LENGTH = 30


def load_yolo():
    """
    YOLOv3 모델과 COCO 클래스명을 로드한다.

    OpenCV의 DNN 모듈을 사용하여 사전 훈련된 YOLOv3 네트워크를 읽어온다.
    YOLOv3는 3개의 스케일(13x13, 26x26, 52x52)에서 검출을 수행하므로
    출력 레이어가 3개이다.

    반환: (신경망 객체, 클래스명 리스트, 출력 레이어명 리스트)
    """
    # COCO 데이터셋의 80개 클래스명을 파일에서 읽어온다
    with open(COCO_NAMES, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # OpenCV DNN 모듈로 YOLOv3 네트워크를 로드한다
    # readNet은 weights(가중치)와 cfg(구조 설정) 파일을 인자로 받는다
    net = cv.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)

    # 출력 레이어 이름을 추출한다
    # getUnconnectedOutLayers()는 출력 레이어의 인덱스(1-based)를 반환한다
    layer_names = net.getLayerNames()
    output_layers = [
        layer_names[i - 1] for i in net.getUnconnectedOutLayers()
    ]

    print(f"[INFO] YOLOv3 모델 로드 완료 - {len(classes)}개 클래스")
    return net, classes, output_layers


def detect_objects(frame, net, output_layers, classes):
    """
    YOLOv3로 한 프레임에서 객체를 검출한다.

    처리 과정:
      1. 이미지를 blob으로 변환 (정규화 + 리사이즈 + BGR->RGB)
      2. 네트워크 순전파(forward) 실행
      3. 신뢰도 임계값 이상인 검출만 수집
      4. NMS(Non-Maximum Suppression)로 중복 박스 제거

    반환:
      detections: [[x1, y1, x2, y2, confidence], ...] numpy 배열
      det_class_ids: 각 검출에 대응하는 클래스 ID 리스트
    """
    h, w = frame.shape[:2]

    # 이미지를 YOLOv3 입력 형태(blob)로 변환한다
    # scalefactor=1/255: 픽셀값을 0~1로 정규화
    # size=(416,416): YOLOv3 고정 입력 크기
    # swapRB=True: OpenCV의 BGR을 네트워크가 요구하는 RGB로 변환
    blob = cv.dnn.blobFromImage(
        frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
        swapRB=True, crop=False
    )
    net.setInput(blob)

    # 순전파(forward) 실행하여 3개 스케일의 검출 결과를 얻는다
    outputs = net.forward(output_layers)

    # 검출 결과를 파싱한다
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            # detection 구조: [cx, cy, bw, bh, objectness, class0_prob, class1_prob, ...]
            # detection[5:] = 80개 COCO 클래스별 확률
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # 신뢰도가 임계값 이상인 검출만 수집한다
            if confidence > CONF_THRESHOLD:
                # YOLO 출력: 중심좌표(cx, cy)와 너비/높이(bw, bh) (전부 정규화된 값)
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)

                # 좌상단(x1, y1) 좌표로 변환한다
                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)

                boxes.append([x1, y1, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # NMS(Non-Maximum Suppression)로 겹치는 박스 중 가장 신뢰도 높은 것만 남긴다
    indices = cv.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    # SORT가 요구하는 형식으로 변환: [x1, y1, x2, y2, score]
    detections = []
    det_class_ids = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            detections.append([x, y, x + w_box, y + h_box, confidences[i]])
            det_class_ids.append(class_ids[i])

    return np.array(detections), det_class_ids


def compute_iou(box1, box2):
    """
    두 바운딩 박스 간의 IoU(Intersection over Union)를 계산한다.
    입력: box1, box2 = [x1, y1, x2, y2]

    IoU는 두 박스의 교집합 영역을 합집합 영역으로 나눈 값으로,
    1.0이면 완전히 겹치고, 0.0이면 전혀 겹치지 않는다.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union


def get_color(track_id):
    """
    트랙 ID별로 고유한 색상을 생성한다 (HSV -> BGR 변환).
    소수 37을 곱해서 인접 ID끼리도 시각적으로 잘 구분되는 색상을 만든다.
    """
    hue = int((track_id * 37) % 180)
    color_hsv = np.uint8([[[hue, 255, 200]]])
    color_bgr = cv.cvtColor(color_hsv, cv.COLOR_HSV2BGR)
    return int(color_bgr[0, 0, 0]), int(color_bgr[0, 0, 1]), int(color_bgr[0, 0, 2])


def main():
    """
    메인 실행 함수

    전체 파이프라인:
      1. YOLOv3 모델 로드
      2. SORT 추적기 초기화
      3. 비디오 프레임 루프:
         a. YOLOv3로 객체 검출
         b. SORT 추적기 업데이트 (칼만 필터 예측 + 헝가리안 매칭)
         c. 추적 결과 시각화 (경계상자 + ID + 클래스명 + 궤적)
      4. 결과 비디오 저장
    """
    print("=" * 60)
    print("과제 01: SORT 알고리즘을 활용한 다중 객체 추적기")
    print("=" * 60)

    # ── 1. YOLOv3 모델 로드 ──
    net, classes, output_layers = load_yolo()

    # ── 2. SORT 추적기 초기화 ──
    # max_age=30: 30프레임 동안 검출 없어도 트랙 유지 (단기 가림 대응)
    # min_hits=3: 3프레임 연속 검출되어야 트랙을 출력에 포함 (잡음 필터링)
    # iou_threshold=0.3: IoU 0.3 이상이어야 같은 객체로 매칭
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # 트랙별 클래스명을 저장하는 딕셔너리 (track_id -> class_name)
    track_classes = {}

    # 트랙별 이동 궤적을 저장하는 딕셔너리 (track_id -> deque of center points)
    # KLT 추적(프로그램 10-2)의 mask 누적 방식과 유사한 개념이다
    track_trajectories = defaultdict(lambda: deque(maxlen=TRAJECTORY_LENGTH))

    # ── 3. 입력 비디오 열기 ──
    cap = cv.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] 비디오 파일을 열 수 없습니다: {VIDEO_PATH}")
        return

    # 비디오 속성 읽기
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] 비디오: {width}x{height}, {fps}FPS, 총 {total_frames}프레임")

    # ── 4. 결과 비디오 저장용 VideoWriter 설정 ──
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_num = 0
    prev_time = time.time()

    # ── 5. 프레임 처리 루프 ──
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # ────────────────────────────────────
        # 단계 1: YOLOv3로 객체 검출
        # ────────────────────────────────────
        detections, det_class_ids = detect_objects(frame, net, output_layers, classes)

        # ────────────────────────────────────
        # 단계 2: SORT 추적기 업데이트
        # ────────────────────────────────────
        # sort.update() 한 줄로 내부에서 칼만 필터 예측과 헝가리안 매칭이 처리된다
        # 입력: [[x1, y1, x2, y2, score], ...]
        # 출력: [[x1, y1, x2, y2, track_id], ...]
        if len(detections) > 0:
            tracks = tracker.update(detections)
        else:
            # 검출이 없는 프레임에서도 반드시 빈 배열로 호출해야 한다
            # 이때 SORT 내부에서는 칼만 필터 예측만 수행한다
            tracks = tracker.update(np.empty((0, 5)))

        # ────────────────────────────────────
        # 단계 3: 트랙-검출 클래스 매핑
        # ────────────────────────────────────
        # SORT는 클래스 정보를 반환하지 않으므로,
        # 추적 결과 박스와 검출 박스의 IoU를 비교하여 클래스명을 부여한다
        if len(detections) > 0 and len(tracks) > 0:
            for track in tracks:
                tid = int(track[4])
                track_box = track[:4]

                # 검출 박스 중 IoU가 가장 높은 것의 클래스를 할당한다
                best_iou = 0
                best_class_id = -1
                for det, cls_id in zip(detections, det_class_ids):
                    iou = compute_iou(track_box, det[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_class_id = cls_id

                if best_class_id >= 0:
                    track_classes[tid] = classes[best_class_id]

        # ────────────────────────────────────
        # 단계 4: 추적 결과 시각화
        # ────────────────────────────────────
        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            color = get_color(track_id)

            # 바운딩 박스 중심점 계산하여 궤적에 저장
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            track_trajectories[track_id].append((cx, cy))

            # (a) 이동 궤적 그리기 (최근 N개 위치를 선으로 연결)
            # KLT 추적의 mask 누적 방식처럼, 과거 위치를 이어서 그린다
            pts = list(track_trajectories[track_id])
            for j in range(1, len(pts)):
                # 궤적이 오래될수록 투명하게 (두께를 줄여서 표현)
                thickness = max(1, int(2 * (j / len(pts))))
                cv.line(frame, pts[j - 1], pts[j], color, thickness)

            # (b) 경계 상자 그리기
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # (c) ID + 클래스명 라벨 표시
            class_name = track_classes.get(track_id, "object")
            label = f"ID:{track_id} {class_name}"
            label_size, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # 라벨 배경 사각형 (글자가 잘 보이도록)
            cv.rectangle(
                frame,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0] + 4, y1),
                color, -1
            )
            # 라벨 텍스트 (흰색)
            cv.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        # FPS 계산
        curr_time = time.time()
        fps_display = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # 화면 상단에 프레임 정보 표시
        info_text = f"Frame: {frame_num}/{total_frames} | Tracks: {len(tracks)} | FPS: {fps_display:.1f}"
        # 텍스트 배경 (가독성 향상)
        cv.rectangle(frame, (0, 0), (450, 35), (0, 0, 0), -1)
        cv.putText(
            frame, info_text, (10, 25),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # 결과 비디오에 프레임 기록
        out.write(frame)

        # 실시간 화면 출력
        cv.imshow("SORT Multi-Object Tracking", frame)

        # 'q' 또는 ESC 키로 종료
        key = cv.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("[INFO] 사용자 종료")
            break

        # 진행 상황을 50프레임마다 출력한다
        if frame_num % 50 == 0:
            print(f"  [{frame_num}/{total_frames}] 추적 중... (현재 {len(tracks)}개 객체)")

    # ── 6. 자원 해제 ──
    cap.release()
    out.release()
    cv.destroyAllWindows()

    print(f"\n[완료] 총 {frame_num}프레임 처리")
    print(f"[저장] {OUTPUT_VIDEO}")
    print("=" * 60)


if __name__ == "__main__":
    main()
