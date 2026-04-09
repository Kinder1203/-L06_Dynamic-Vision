import cv2 as cv
import numpy as np
import os
import time
from collections import defaultdict, deque
from sort import Sort

# 현재 스크립트가 위치한 디렉터리를 기준으로 모든 경로를 설정한다
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 입력 파일(비디오, 모델)이 있는 폴더 경로를 설정한다
INPUT_DIR = os.path.join(BASE_DIR, "input_files")
# 결과 파일을 저장할 폴더 경로를 설정한다
RESULT_DIR = os.path.join(BASE_DIR, "result")
# 결과 폴더가 없으면 생성하고, 이미 존재하면 에러 없이 넘어간다
os.makedirs(RESULT_DIR, exist_ok=True)

# 입력 비디오 경로를 설정한다
VIDEO_PATH = os.path.join(INPUT_DIR, "slow_traffic_small.mp4")

# YOLOv3 모델 가중치 파일 경로를 설정한다
YOLO_WEIGHTS = os.path.join(INPUT_DIR, "yolov3.weights")
# YOLOv3 모델 구조 설정 파일 경로를 설정한다
YOLO_CFG = os.path.join(INPUT_DIR, "yolov3.cfg")
# COCO 데이터셋 80개 클래스명 파일 경로를 설정한다
COCO_NAMES = os.path.join(BASE_DIR, "coco.names")

# 결과 비디오 저장 경로를 설정한다
OUTPUT_VIDEO = os.path.join(RESULT_DIR, "01_sort_tracking.mp4")

# YOLOv3 설정 상수
# 검출 신뢰도 임계값을 설정한다 (이 값 이상만 검출로 인정)
CONF_THRESHOLD = 0.5
# Non-Maximum Suppression 임계값을 설정한다 (중복 박스 제거 기준)
NMS_THRESHOLD = 0.4
# YOLOv3 입력 해상도를 설정한다 (416x416 고정)
INPUT_SIZE = 416

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
        # 각 줄의 앞뒤 공백을 제거하여 클래스명 리스트를 만든다
        classes = [line.strip() for line in f.readlines()]

    # OpenCV DNN 모듈로 YOLOv3 네트워크를 로드한다
    # readNet은 weights(가중치)와 cfg(구조 설정) 파일을 인자로 받는다
    net = cv.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)

    # 전체 레이어 이름 목록을 가져온다
    layer_names = net.getLayerNames()
    # 출력 레이어(연결되지 않은 레이어)의 인덱스를 가져와 이름으로 변환한다
    # getUnconnectedOutLayers()는 1-based 인덱스를 반환하므로 -1 해준다
    output_layers = [
        layer_names[i - 1] for i in net.getUnconnectedOutLayers()
    ]

    # 모델 로드 완료 메시지를 출력한다
    print(f"[INFO] YOLOv3 모델 로드 완료 - {len(classes)}개 클래스")
    # 신경망 객체, 클래스명, 출력 레이어명을 반환한다
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
    # 프레임의 높이(h)와 너비(w)를 가져온다
    h, w = frame.shape[:2]

    # 이미지를 YOLOv3 입력 형태(blob)로 변환한다
    # scalefactor=1/255: 픽셀값을 0~1로 정규화
    # size=(416,416): YOLOv3 고정 입력 크기
    # swapRB=True: OpenCV의 BGR을 네트워크가 요구하는 RGB로 변환
    blob = cv.dnn.blobFromImage(
        frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
        swapRB=True, crop=False
    )
    # 변환된 blob을 네트워크 입력으로 설정한다
    net.setInput(blob)

    # 순전파(forward)를 실행하여 3개 스케일의 검출 결과를 얻는다
    outputs = net.forward(output_layers)

    # 검출 결과를 저장할 빈 리스트들을 초기화한다
    boxes = []          # 바운딩 박스 좌표 리스트
    confidences = []    # 신뢰도 리스트
    class_ids = []      # 클래스 ID 리스트

    # 3개 스케일의 출력을 순회한다
    for output in outputs:
        # 각 출력의 검출 결과를 순회한다
        for detection in output:
            # detection[5:] = 80개 COCO 클래스별 확률을 추출한다
            scores = detection[5:]
            # 확률이 가장 높은 클래스의 인덱스를 구한다
            class_id = np.argmax(scores)
            # 해당 클래스의 신뢰도(확률)를 가져온다
            confidence = scores[class_id]

            # 신뢰도가 임계값 이상인 검출만 수집한다
            if confidence > CONF_THRESHOLD:
                # YOLO 출력: 정규화된 중심좌표(cx, cy)를 픽셀 좌표로 변환한다
                cx = int(detection[0] * w)
                # 정규화된 중심 y좌표를 픽셀 좌표로 변환한다
                cy = int(detection[1] * h)
                # 정규화된 너비를 픽셀 단위로 변환한다
                bw = int(detection[2] * w)
                # 정규화된 높이를 픽셀 단위로 변환한다
                bh = int(detection[3] * h)

                # 중심좌표에서 좌상단(x1, y1) 좌표로 변환한다
                x1 = int(cx - bw / 2)
                # 중심좌표에서 좌상단 y 좌표로 변환한다
                y1 = int(cy - bh / 2)

                # 바운딩 박스 좌표를 리스트에 추가한다
                boxes.append([x1, y1, bw, bh])
                # 신뢰도를 float으로 변환하여 리스트에 추가한다
                confidences.append(float(confidence))
                # 클래스 ID를 리스트에 추가한다
                class_ids.append(class_id)

    # NMS(Non-Maximum Suppression)로 겹치는 박스 중 가장 신뢰도 높은 것만 남긴다
    indices = cv.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    # SORT가 요구하는 형식으로 변환할 빈 리스트를 초기화한다
    detections = []      # [x1, y1, x2, y2, score] 형식
    det_class_ids = []   # 각 검출의 클래스 ID

    # NMS 결과가 있으면 처리한다
    if len(indices) > 0:
        # NMS를 통과한 인덱스를 순회한다
        for i in indices.flatten():
            # 해당 인덱스의 바운딩 박스 좌표를 가져온다
            x, y, w_box, h_box = boxes[i]
            # [x1, y1, x2, y2, score] 형식으로 변환하여 추가한다
            detections.append([x, y, x + w_box, y + h_box, confidences[i]])
            # 해당 클래스 ID를 추가한다
            det_class_ids.append(class_ids[i])

    # 검출 결과를 numpy 배열로 변환하여 클래스 ID와 함께 반환한다
    return np.array(detections), det_class_ids


def compute_iou(box1, box2):
    """
    두 바운딩 박스 간의 IoU(Intersection over Union)를 계산한다.
    입력: box1, box2 = [x1, y1, x2, y2]

    IoU는 두 박스의 교집합 영역을 합집합 영역으로 나눈 값으로,
    1.0이면 완전히 겹치고, 0.0이면 전혀 겹치지 않는다.
    """
    # 교집합 영역의 좌상단 x좌표를 구한다 (두 박스 중 더 큰 값)
    x1 = max(box1[0], box2[0])
    # 교집합 영역의 좌상단 y좌표를 구한다 (두 박스 중 더 큰 값)
    y1 = max(box1[1], box2[1])
    # 교집합 영역의 우하단 x좌표를 구한다 (두 박스 중 더 작은 값)
    x2 = min(box1[2], box2[2])
    # 교집합 영역의 우하단 y좌표를 구한다 (두 박스 중 더 작은 값)
    y2 = min(box1[3], box2[3])

    # 교집합 면적을 계산한다 (겹치지 않으면 0)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    # 첫 번째 박스의 면적을 계산한다
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    # 두 번째 박스의 면적을 계산한다
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # 합집합 면적을 계산한다 (두 면적의 합에서 교집합을 뺀다)
    union = area1 + area2 - intersection

    # 합집합이 0이면 (두 박스 모두 크기가 0) IoU를 0.0으로 반환한다
    if union == 0:
        return 0.0
    # IoU 값을 계산하여 반환한다
    return intersection / union


def get_color(track_id):
    """
    트랙 ID별로 고유한 색상을 생성한다 (HSV -> BGR 변환).
    소수 37을 곱해서 인접 ID끼리도 시각적으로 잘 구분되는 색상을 만든다.
    """
    # ID에 소수 37을 곱하고 180으로 나머지를 구해 색상환(Hue) 값을 만든다
    hue = int((track_id * 37) % 180)
    # HSV 색상을 1x1 이미지로 만든다 (Hue, Saturation=255, Value=200)
    color_hsv = np.uint8([[[hue, 255, 200]]])
    # HSV를 BGR로 변환한다 (OpenCV는 BGR을 사용하므로)
    color_bgr = cv.cvtColor(color_hsv, cv.COLOR_HSV2BGR)
    # BGR 채널 값을 각각 정수로 추출하여 튜플로 반환한다
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
    # 구분선을 출력한다
    print("=" * 60)
    # 과제 제목을 출력한다
    print("과제 01: SORT 알고리즘을 활용한 다중 객체 추적기")
    # 구분선을 출력한다
    print("=" * 60)

    # ── 1. YOLOv3 모델 로드 ──
    # YOLOv3 신경망, 클래스명, 출력 레이어를 로드한다
    net, classes, output_layers = load_yolo()

    # ── 2. SORT 추적기 초기화 ──
    # max_age=30: 30프레임 동안 검출 없어도 트랙 유지 (단기 가림 대응)
    # min_hits=3: 3프레임 연속 검출되어야 트랙을 출력에 포함 (잡음 필터링)
    # iou_threshold=0.3: IoU 0.3 이상이어야 같은 객체로 매칭
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # 트랙별 클래스명을 저장하는 딕셔너리를 초기화한다 (track_id -> class_name)
    track_classes = {}

    # 트랙별 이동 궤적을 저장하는 딕셔너리를 초기화한다 (track_id -> deque of center points)
    track_trajectories = defaultdict(lambda: deque(maxlen=TRAJECTORY_LENGTH))

    # ── 3. 입력 비디오 열기 ──
    # 비디오 파일을 열어 VideoCapture 객체를 생성한다
    cap = cv.VideoCapture(VIDEO_PATH)
    # 비디오를 열 수 없으면 에러를 출력하고 종료한다
    if not cap.isOpened():
        # 에러 메시지를 출력한다
        print(f"[ERROR] 비디오 파일을 열 수 없습니다: {VIDEO_PATH}")
        # 함수를 조기 종료한다
        return

    # 비디오의 FPS(초당 프레임 수)를 정수로 가져온다
    fps = int(cap.get(cv.CAP_PROP_FPS))
    # 비디오의 가로 해상도를 정수로 가져온다
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # 비디오의 세로 해상도를 정수로 가져온다
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # 비디오의 총 프레임 수를 정수로 가져온다
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # 비디오 정보를 출력한다
    print(f"[INFO] 비디오: {width}x{height}, {fps}FPS, 총 {total_frames}프레임")

    # ── 4. 결과 비디오 저장용 VideoWriter 설정 ──
    # mp4v 코덱을 4자리 문자로 설정한다
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # VideoWriter 객체를 생성한다 (출력경로, 코덱, FPS, 해상도)
    out = cv.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # 프레임 카운터를 0으로 초기화한다
    frame_num = 0
    # FPS 계산을 위한 이전 시간을 현재 시간으로 기록한다
    prev_time = time.time()

    # ── 5. 프레임 처리 루프 ──
    # 무한 루프로 매 프레임을 처리한다
    while True:
        # 비디오에서 한 프레임을 읽어온다
        ret, frame = cap.read()
        # 더 이상 읽을 프레임이 없으면 루프를 종료한다
        if not ret:
            # 루프를 탈출한다
            break

        # 프레임 번호를 1 증가시킨다
        frame_num += 1

        # ────────────────────────────────────
        # 단계 1: YOLOv3로 객체 검출
        # ────────────────────────────────────
        # 현재 프레임에서 객체를 검출하고 클래스 ID를 함께 반환받는다
        detections, det_class_ids = detect_objects(frame, net, output_layers, classes)

        # ────────────────────────────────────
        # 단계 2: SORT 추적기 업데이트
        # ────────────────────────────────────
        # sort.update() 한 줄로 내부에서 칼만 필터 예측과 헝가리안 매칭이 처리된다
        # 입력: [[x1, y1, x2, y2, score], ...]
        # 출력: [[x1, y1, x2, y2, track_id], ...]
        # 검출 결과가 있으면 그대로 전달한다
        if len(detections) > 0:
            # 검출 결과를 SORT에 전달하여 추적 결과를 받는다
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
            # 각 추적 결과를 순회한다
            for track in tracks:
                # 트랙 ID를 정수로 변환한다
                tid = int(track[4])
                # 트랙의 바운딩 박스 좌표를 가져온다
                track_box = track[:4]

                # 검출 박스 중 IoU가 가장 높은 것을 찾기 위한 변수를 초기화한다
                best_iou = 0
                best_class_id = -1
                # 모든 검출 결과와 IoU를 비교한다
                for det, cls_id in zip(detections, det_class_ids):
                    # 트랙 박스와 검출 박스의 IoU를 계산한다
                    iou = compute_iou(track_box, det[:4])
                    # 현재까지의 최고 IoU보다 높으면 갱신한다
                    if iou > best_iou:
                        # 최고 IoU 값을 갱신한다
                        best_iou = iou
                        # 최고 IoU에 해당하는 클래스 ID를 저장한다
                        best_class_id = cls_id

                # 유효한 클래스가 매칭되었으면 딕셔너리에 저장한다
                if best_class_id >= 0:
                    # 트랙 ID에 클래스명을 매핑한다
                    track_classes[tid] = classes[best_class_id]

        # ────────────────────────────────────
        # 단계 4: 추적 결과 시각화
        # ────────────────────────────────────
        # 각 추적 결과를 순회하여 시각화한다
        for track in tracks:
            # 추적 결과에서 좌표와 트랙 ID를 정수로 변환한다
            x1, y1, x2, y2, track_id = track.astype(int)
            # 트랙 ID에 해당하는 고유 색상을 생성한다
            color = get_color(track_id)

            # 바운딩 박스의 중심점 x좌표를 계산한다
            cx = (x1 + x2) // 2
            # 바운딩 박스의 중심점 y좌표를 계산한다
            cy = (y1 + y2) // 2
            # 중심점을 궤적 deque에 추가한다
            track_trajectories[track_id].append((cx, cy))

            # (a) 이동 궤적 그리기 (최근 N개 위치를 선으로 연결)
            # 궤적 deque를 리스트로 변환한다
            pts = list(track_trajectories[track_id])
            # 연속된 점들을 선으로 연결한다
            for j in range(1, len(pts)):
                # 궤적이 최근일수록 두껍게 그린다
                thickness = max(1, int(2 * (j / len(pts))))
                # 이전 점과 현재 점을 선으로 연결한다
                cv.line(frame, pts[j - 1], pts[j], color, thickness)

            # (b) 경계 상자를 그린다 (두께 2)
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # (c) ID + 클래스명 라벨 표시
            # 트랙 ID에 해당하는 클래스명을 가져온다 (없으면 "object")
            class_name = track_classes.get(track_id, "object")
            # 라벨 텍스트를 구성한다
            label = f"ID:{track_id} {class_name}"
            # 라벨 텍스트의 크기를 계산한다
            label_size, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # 라벨 배경 사각형을 채워서 글자가 잘 보이도록 한다
            cv.rectangle(
                frame,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0] + 4, y1),
                color, -1
            )
            # 라벨 텍스트를 흰색으로 출력한다
            cv.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        # 현재 시간을 가져와 FPS를 계산한다
        curr_time = time.time()
        # 이전 프레임과의 시간 차이로 FPS 값을 계산한다
        fps_display = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        # 현재 시간을 이전 시간으로 갱신한다
        prev_time = curr_time

        # 화면 상단에 표시할 정보 텍스트를 구성한다
        info_text = f"Frame: {frame_num}/{total_frames} | Tracks: {len(tracks)} | FPS: {fps_display:.1f}"
        # 텍스트 배경을 검은색 사각형으로 깔아서 가독성을 높인다
        cv.rectangle(frame, (0, 0), (450, 35), (0, 0, 0), -1)
        # 정보 텍스트를 초록색으로 화면에 출력한다
        cv.putText(
            frame, info_text, (10, 25),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # 현재 프레임을 결과 비디오 파일에 기록한다
        out.write(frame)

        # 실시간으로 화면에 프레임을 출력한다
        cv.imshow("SORT Multi-Object Tracking", frame)

        # 1ms 동안 키 입력을 대기하고, 하위 8비트만 추출한다
        key = cv.waitKey(1) & 0xFF
        # 'q' 키 또는 ESC 키가 눌리면 프로그램을 종료한다
        if key == ord("q") or key == 27:
            # 종료 메시지를 출력한다
            print("[INFO] 사용자 종료")
            # 루프를 탈출한다
            break

        # 50프레임마다 진행 상황을 출력한다
        if frame_num % 50 == 0:
            # 현재 프레임 번호와 추적 중인 객체 수를 출력한다
            print(f"  [{frame_num}/{total_frames}] 추적 중... (현재 {len(tracks)}개 객체)")

    # ── 6. 자원 해제 ──
    # 비디오 캡처 장치를 해제한다
    cap.release()
    # 비디오 파일 쓰기를 완료하고 파일을 닫는다
    out.release()
    # 모든 OpenCV 창을 닫는다
    cv.destroyAllWindows()

    # 처리 완료 메시지를 출력한다
    print(f"\n[완료] 총 {frame_num}프레임 처리")
    # 저장된 결과 비디오 경로를 출력한다
    print(f"[저장] {OUTPUT_VIDEO}")
    # 구분선을 출력한다
    print("=" * 60)


# 이 파일이 직접 실행될 때만 main 함수를 호출한다
if __name__ == "__main__":
    # main 함수를 실행한다
    main()
