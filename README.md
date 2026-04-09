# 컴퓨터 비전 비디오 분석 실습 (SORT 다중 객체 추적, MediaPipe FaceMesh)

컴퓨터 비전 수업의 6번째 실습 과제 저장소입니다.  
이번 과제는 YOLOv3 객체 검출기와 SORT 알고리즘을 결합한 다중 객체 추적기 구현, 그리고 MediaPipe의 FaceMesh 모듈을 활용한 실시간 얼굴 랜드마크 추출 및 시각화를 목표로 합니다.  
강의에서 배운 추적(Tracking)의 개념(칼만 필터, 헝가리안 알고리즘, IoU 기반 데이터 연관), MediaPipe를 이용한 사람 인식(FaceMesh 468개 랜드마크)을 그대로 반영하여 구현했습니다.

---

## 환경 설정

| 항목 | 버전/도구 |
|---|---|
| Python | 3.10 |
| 주요 라이브러리 | `opencv-python`, `mediapipe`, `filterpy`, `scipy`, `numpy` |
| 입력 파일 폴더 | `input_files/` |
| 결과 파일 폴더 | `result/` |

```bash
pip install opencv-python mediapipe filterpy scipy numpy
```

## 실행 방법

프로젝트 루트(`computer_vison/6/`)에서 터미널을 열고 다음 명령어를 실행합니다.

```bash
python 01_sort_tracking.py
python 02_facemesh_landmark.py
```

> 참고: 과제 01은 입력 비디오(`slow_traffic_small.mp4`)와 YOLOv3 모델 파일(`yolov3.weights`, `yolov3.cfg`)이 `input_files/` 폴더에 있어야 합니다.  
> 과제 02는 웹캠이 필요하며, ESC 키를 누르면 종료되고 결과 비디오가 저장됩니다.

---

## 실습 01 — SORT 알고리즘을 활용한 다중 객체 추적기 구현

### 1. 과제에 대한 설명
SORT(Simple Online and Realtime Tracking) 알고리즘을 사용하여 비디오에서 다중 객체를 실시간으로 추적하는 프로그램을 구현합니다.
- OpenCV의 DNN 모듈을 사용하여 사전 훈련된 YOLOv3 모델을 로드하고, 각 프레임에서 객체를 검출합니다.
- 검출된 객체의 경계상자를 입력으로 받아 SORT 추적기를 초기화합니다.
- SORT 알고리즘은 **칼만 필터**로 객체의 다음 위치를 예측하고, **헝가리안 알고리즘**으로 예측 박스와 검출 박스 간의 최적 매칭을 수행하여 추적을 유지합니다.
- 추적된 각 객체에 고유 ID를 부여하고, ID와 경계상자를 비디오 프레임에 표시하여 실시간으로 출력합니다.
- 결과를 `result/01_sort_tracking.mp4`로 저장합니다.

### 2. 핵심 코드 설명
```python
# [핵심] SORT 추적기 초기화 - 칼만 필터 + 헝가리안 알고리즘 기반 다중 객체 추적기
# max_age=30: 30프레임 동안 검출 없어도 트랙 유지, min_hits=3: 3프레임 연속 검출 시 출력
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# [핵심] YOLOv3로 객체 검출 - OpenCV DNN 모듈 사용
blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)

# [핵심] SORT 추적기 업데이트 - 이 한 줄로 칼만 필터 예측 + 헝가리안 매칭이 처리된다
# 입력: [[x1, y1, x2, y2, score], ...], 출력: [[x1, y1, x2, y2, track_id], ...]
tracks = tracker.update(detections)

# [핵심] 추적 결과 시각화 - 고유 ID와 경계상자를 프레임에 표시
for track in tracks:
    x1, y1, x2, y2, track_id = track.astype(int)
    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv.putText(frame, f"ID:{track_id} {class_name}", (x1, y1-5), ...)
```

### 3. 전체 코드

#### 3-1. sort.py (SORT 알고리즘 모듈)

`01_sort_tracking.py`가 내부적으로 import하여 사용하는 SORT 추적 모듈입니다.  
칼만 필터(`filterpy`)와 헝가리안 알고리즘(`scipy`)을 사용하여 다중 객체를 추적합니다.

```python
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    """
    헝가리안 알고리즘을 사용하여 비용 행렬(cost matrix)에서
    전체 비용이 최소가 되는 최적의 1:1 매칭을 수행합니다.
    """
    # scipy의 linear_sum_assignment로 최적 할당 문제를 풀어 행/열 인덱스를 반환한다
    x, y = linear_sum_assignment(cost_matrix)
    # (행, 열) 쌍을 numpy 배열로 변환하여 반환한다
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    두 바운딩 박스 집합 간의 IoU(Intersection over Union)를 일괄 계산합니다.
    입력 형식: [x1, y1, x2, y2]
    """
    # 브로드캐스팅을 위해 gt 박스 배열에 차원을 추가한다
    bb_gt = np.expand_dims(bb_gt, 0)
    # 브로드캐스팅을 위해 test 박스 배열에 차원을 추가한다
    bb_test = np.expand_dims(bb_test, 1)

    # 교집합 영역의 좌상단 x좌표를 구한다 (두 값 중 큰 값)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    # 교집합 영역의 좌상단 y좌표를 구한다 (두 값 중 큰 값)
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    # 교집합 영역의 우하단 x좌표를 구한다 (두 값 중 작은 값)
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    # 교집합 영역의 우하단 y좌표를 구한다 (두 값 중 작은 값)
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    # 교집합 영역의 너비를 구한다 (음수면 0으로 클리핑)
    w = np.maximum(0.0, xx2 - xx1)
    # 교집합 영역의 높이를 구한다 (음수면 0으로 클리핑)
    h = np.maximum(0.0, yy2 - yy1)
    # 교집합 면적을 계산한다
    wh = w * h

    # IoU = 교집합 면적 / 합집합 면적 으로 계산한다
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    # IoU 행렬을 반환한다
    return o


def convert_bbox_to_z(bbox):
    """
    바운딩 박스 [x1, y1, x2, y2]를 칼만 필터 상태 벡터 [x, y, s, r]로 변환합니다.
    x, y: 중심 좌표 / s: 면적(scale) / r: 종횡비(aspect ratio)
    """
    # 바운딩 박스의 너비를 계산한다
    w = bbox[2] - bbox[0]
    # 바운딩 박스의 높이를 계산한다
    h = bbox[3] - bbox[1]
    # 바운딩 박스의 중심 x좌표를 계산한다
    x = bbox[0] + w / 2.0
    # 바운딩 박스의 중심 y좌표를 계산한다
    y = bbox[1] + h / 2.0
    # 면적(scale)을 너비 x 높이로 계산한다
    s = w * h
    # 종횡비(aspect ratio)를 너비 / 높이로 계산한다
    r = w / float(h)
    # [x, y, s, r] 형태의 (4,1) 열벡터로 반환한다
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    칼만 필터 상태 벡터 [x, y, s, r]을 바운딩 박스 [x1, y1, x2, y2]로 역변환합니다.
    """
    # 면적(s)과 종횡비(r)로부터 너비를 복원한다
    w = np.sqrt(x[2] * x[3])
    # 면적(s)를 너비로 나누어 높이를 복원한다
    h = x[2] / w
    # score가 없으면 [x1, y1, x2, y2] 형태의 (1,4) 배열로 반환한다
    if score is None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    # score가 있으면 [x1, y1, x2, y2, score] 형태의 (1,5) 배열로 반환한다
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


class KalmanBoxTracker:
    """
    개별 추적 대상의 내부 상태를 칼만 필터로 관리하는 클래스입니다.
    상태 벡터: [x, y, s, r, dx, dy, ds] - 위치, 면적, 종횡비 + 각각의 속도
    """

    # 전체 트래커에서 고유 ID를 부여하기 위한 클래스 변수
    count = 0

    def __init__(self, bbox):
        """초기 바운딩 박스로 칼만 필터 트래커를 생성합니다."""
        # 등속 운동 모델(constant velocity model)로 칼만 필터를 정의한다
        # 상태 차원 7 (x, y, s, r, dx, dy, ds), 관측 차원 4 (x, y, s, r)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # 상태 전이 행렬 F: 현재 상태에서 다음 상태를 예측하는 행렬
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x' = x + dx
            [0, 1, 0, 0, 0, 1, 0],  # y' = y + dy
            [0, 0, 1, 0, 0, 0, 1],  # s' = s + ds
            [0, 0, 0, 1, 0, 0, 0],  # r' = r (종횡비는 속도 없음)
            [0, 0, 0, 0, 1, 0, 0],  # dx' = dx (등속)
            [0, 0, 0, 0, 0, 1, 0],  # dy' = dy (등속)
            [0, 0, 0, 0, 0, 0, 1],  # ds' = ds (등속)
        ])

        # 관측 행렬 H: 상태 벡터에서 관측 가능한 부분(x, y, s, r)만 추출한다
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # x 관측
            [0, 1, 0, 0, 0, 0, 0],  # y 관측
            [0, 0, 1, 0, 0, 0, 0],  # s 관측
            [0, 0, 0, 1, 0, 0, 0],  # r 관측
        ])

        # 관측 노이즈 공분산 행렬 R의 면적/종횡비 관련 값을 크게 설정한다
        self.kf.R[2:, 2:] *= 10.0
        # 초기 상태 공분산 P의 속도 관련 값에 높은 불확실성을 부여한다
        self.kf.P[4:, 4:] *= 1000.0
        # 전체 초기 불확실성을 10배로 키운다
        self.kf.P *= 10.0
        # 프로세스 노이즈 Q의 마지막 요소(ds 변화율)를 작게 설정한다
        self.kf.Q[-1, -1] *= 0.01
        # 프로세스 노이즈 Q의 속도 관련 요소를 작게 설정한다
        self.kf.Q[4:, 4:] *= 0.01

        # 초기 바운딩 박스를 상태 벡터의 위치 부분에 설정한다
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        # 마지막 업데이트 이후 경과 프레임 수를 0으로 초기화한다
        self.time_since_update = 0
        # 고유 트랙 ID를 클래스 변수에서 할당한다
        self.id = KalmanBoxTracker.count
        # 다음 트래커를 위해 클래스 변수를 1 증가시킨다
        KalmanBoxTracker.count += 1
        # 예측 이력을 저장하는 빈 리스트를 초기화한다
        self.history = []
        # 총 매칭 횟수를 0으로 초기화한다
        self.hits = 0
        # 연속 매칭 횟수를 0으로 초기화한다
        self.hit_streak = 0
        # 트래커 생성 이후 총 경과 프레임 수를 0으로 초기화한다
        self.age = 0

    def update(self, bbox):
        """관측된 바운딩 박스로 칼만 필터 상태를 보정(update)합니다."""
        # 업데이트가 이루어졌으므로 경과 시간을 0으로 리셋한다
        self.time_since_update = 0
        # 예측 이력을 비운다
        self.history = []
        # 총 매칭 횟수를 1 증가시킨다
        self.hits += 1
        # 연속 매칭 횟수를 1 증가시킨다
        self.hit_streak += 1
        # 관측값으로 칼만 필터 상태를 보정한다
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """칼만 필터로 다음 프레임의 위치를 예측합니다."""
        # 면적이 음수가 되는 것을 방지한다 (면적 변화율 + 현재 면적이 0 이하이면)
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            # 면적 변화율을 0으로 설정한다
            self.kf.x[6] *= 0.0
        # 칼만 필터로 다음 상태를 예측한다
        self.kf.predict()
        # 트래커 수명을 1 증가시킨다
        self.age += 1
        # 업데이트 없이 예측만 한 경우 연속 매칭을 0으로 리셋한다
        if self.time_since_update > 0:
            # 연속 매칭 횟수를 0으로 리셋한다
            self.hit_streak = 0
        # 마지막 업데이트 이후 경과 프레임 수를 1 증가시킨다
        self.time_since_update += 1
        # 예측된 바운딩 박스를 이력에 저장한다
        self.history.append(convert_x_to_bbox(self.kf.x))
        # 가장 최근 예측 결과를 반환한다
        return self.history[-1]

    def get_state(self):
        """현재 바운딩 박스 추정값을 반환합니다."""
        # 현재 칼만 필터 상태를 바운딩 박스로 변환하여 반환한다
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    검출된 객체와 기존 트래커를 IoU 기반으로 매칭합니다.
    반환: (매칭된 쌍, 미매칭 검출, 미매칭 트래커)
    """
    # 트래커가 하나도 없으면 모든 검출을 미매칭으로 반환한다
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),      # 매칭된 쌍: 없음
            np.arange(len(detections)),         # 미매칭 검출: 전부
            np.empty((0, 5), dtype=int),       # 미매칭 트래커: 없음
        )

    # 검출 박스와 트래커 박스 간의 IoU 비용 행렬을 계산한다
    iou_matrix = iou_batch(detections, trackers)

    # IoU 행렬의 크기가 유효한 경우 매칭을 수행한다
    if min(iou_matrix.shape) > 0:
        # IoU가 임계값을 넘는 위치를 이진 행렬로 변환한다
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # 각 행과 열에서 최대 1개만 임계값을 넘으면 바로 매칭한다 (1:1 확정)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            # where로 1인 위치를 찾아 매칭 인덱스를 만든다
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # 그렇지 않으면 헝가리안 알고리즘으로 최적 매칭을 수행한다 (비용 = -IoU)
            matched_indices = linear_assignment(-iou_matrix)
    else:
        # IoU 행렬이 비어있으면 빈 매칭을 반환한다
        matched_indices = np.empty(shape=(0, 2))

    # 매칭되지 않은 검출 인덱스를 수집한다
    unmatched_detections = []
    # 모든 검출 인덱스를 순회한다
    for d, det in enumerate(detections):
        # 매칭된 인덱스에 포함되지 않으면 미매칭으로 추가한다
        if d not in matched_indices[:, 0]:
            # 미매칭 검출 리스트에 추가한다
            unmatched_detections.append(d)

    # 매칭되지 않은 트래커 인덱스를 수집한다
    unmatched_trackers = []
    # 모든 트래커 인덱스를 순회한다
    for t, trk in enumerate(trackers):
        # 매칭된 인덱스에 포함되지 않으면 미매칭으로 추가한다
        if t not in matched_indices[:, 1]:
            # 미매칭 트래커 리스트에 추가한다
            unmatched_trackers.append(t)

    # IoU가 임계값 미만인 매칭은 무효 처리한다
    matches = []
    # 매칭된 모든 쌍을 순회한다
    for m in matched_indices:
        # 매칭된 쌍의 IoU가 임계값 미만이면 양쪽 모두 미매칭으로 돌린다
        if iou_matrix[m[0], m[1]] < iou_threshold:
            # 검출을 미매칭에 추가한다
            unmatched_detections.append(m[0])
            # 트래커를 미매칭에 추가한다
            unmatched_trackers.append(m[1])
        else:
            # IoU가 충분하면 유효한 매칭으로 추가한다
            matches.append(m.reshape(1, 2))

    # 유효한 매칭이 없으면 빈 배열로 설정한다
    if len(matches) == 0:
        # 빈 (0,2) 배열을 생성한다
        matches = np.empty((0, 2), dtype=int)
    else:
        # 모든 매칭 쌍을 하나의 배열로 합친다
        matches = np.concatenate(matches, axis=0)

    # 매칭 결과, 미매칭 검출, 미매칭 트래커를 반환한다
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    """
    SORT 다중 객체 추적기 메인 클래스입니다.
    매 프레임마다 update()를 호출하면 내부에서 칼만 필터 예측 + 헝가리안 매칭이 처리됩니다.
    """

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        max_age: 검출 없이 트랙을 유지하는 최대 프레임 수
        min_hits: 트랙이 출력되기 위한 최소 연속 매칭 횟수
        iou_threshold: 매칭 판별 IoU 임계값
        """
        # 트랙을 유지하는 최대 프레임 수를 저장한다
        self.max_age = max_age
        # 트랙 출력에 필요한 최소 연속 매칭 횟수를 저장한다
        self.min_hits = min_hits
        # 매칭 판별 IoU 임계값을 저장한다
        self.iou_threshold = iou_threshold
        # 현재 활성 트래커 목록을 빈 리스트로 초기화한다
        self.trackers = []
        # 처리된 프레임 수를 0으로 초기화한다
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        검출 결과를 받아 추적을 수행합니다.

        입력: dets - [[x1, y1, x2, y2, score], ...] 형태의 numpy 배열
        출력: [[x1, y1, x2, y2, track_id], ...] 형태의 numpy 배열

        주의: 검출이 없는 프레임에서도 반드시 빈 배열로 호출해야 합니다.
        """
        # 프레임 카운터를 1 증가시킨다
        self.frame_count += 1

        # 기존 트래커들의 예측 위치를 저장할 배열을 생성한다
        trks = np.zeros((len(self.trackers), 5))
        # 삭제할 트래커 인덱스를 저장할 빈 리스트를 초기화한다
        to_del = []
        # 출력 결과를 저장할 빈 리스트를 초기화한다
        ret = []
        # 각 트래커의 예측 위치를 계산한다
        for t, trk in enumerate(trks):
            # 해당 트래커의 칼만 필터로 다음 위치를 예측한다
            pos = self.trackers[t].predict()[0]
            # 예측된 좌표를 배열에 저장한다
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            # 예측값에 NaN이 있으면 삭제 목록에 추가한다
            if np.any(np.isnan(pos)):
                # 삭제 대상 인덱스를 추가한다
                to_del.append(t)
        # NaN이 포함된 행을 제거하여 유효한 트래커만 남긴다
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # 유효하지 않은 트래커를 역순으로 삭제한다 (인덱스 밀림 방지)
        for t in reversed(to_del):
            # 해당 인덱스의 트래커를 제거한다
            self.trackers.pop(t)

        # 검출-트래커 매칭을 수행한다
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # 매칭된 트래커의 상태를 검출 결과로 업데이트한다
        for m in matched:
            # 매칭된 트래커를 해당 검출 결과로 보정한다
            self.trackers[m[1]].update(dets[m[0], :])

        # 미매칭 검출에 대해 새 트래커를 생성한다
        for i in unmatched_dets:
            # 해당 검출 결과로 새 KalmanBoxTracker를 생성한다
            trk = KalmanBoxTracker(dets[i, :])
            # 트래커 목록에 추가한다
            self.trackers.append(trk)

        # 결과를 수집하고 수명이 다한 트랙을 제거한다
        i = len(self.trackers)
        # 트래커 목록을 역순으로 순회한다 (삭제 시 인덱스 밀림 방지)
        for trk in reversed(self.trackers):
            # 현재 바운딩 박스 추정값을 가져온다
            d = trk.get_state()[0]
            # 최근 업데이트되었고 충분한 매칭 이력이 있으면 결과에 포함한다
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits
                or self.frame_count <= self.min_hits
            ):
                # 바운딩 박스와 ID(+1)를 결합하여 결과에 추가한다
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            # 인덱스를 1 감소시킨다
            i -= 1
            # 마지막 업데이트 이후 max_age 프레임을 초과하면 트랙을 삭제한다
            if trk.time_since_update > self.max_age:
                # 해당 인덱스의 트래커를 제거한다
                self.trackers.pop(i)

        # 결과가 있으면 모든 결과를 합쳐서 반환한다
        if len(ret) > 0:
            return np.concatenate(ret)
        # 결과가 없으면 빈 (0,5) 배열을 반환한다
        return np.empty((0, 5))
```

#### 3-2. 01_sort_tracking.py (메인 실행 코드)

```python
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

        # 단계 1: YOLOv3로 객체 검출
        # 현재 프레임에서 객체를 검출하고 클래스 ID를 함께 반환받는다
        detections, det_class_ids = detect_objects(frame, net, output_layers, classes)

        # 단계 2: SORT 추적기 업데이트
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

        # 단계 3: 트랙-검출 클래스 매핑
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

        # 단계 4: 추적 결과 시각화
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
```

### 4. 최종 결과물

*(SORT 추적 결과 - 비디오의 각 프레임에서 객체가 고유 ID와 경계상자로 추적되며, 이동 궤적이 선으로 표시됩니다.)*

결과 비디오: 

https://github.com/user-attachments/assets/1a6f06b1-358e-4681-afd3-a98f5853c90a

(914프레임 전체 처리 완료)

### 5. 결과 해석
- YOLOv3로 프레임당 객체를 검출하고, SORT 알고리즘의 **칼만 필터 예측 + 헝가리안 매칭**을 통해 각 객체에 고유 ID를 부여하여 안정적으로 추적했습니다.
- `slow_traffic_small.mp4` 비디오(640x360, 29FPS, 914프레임)에서 차량 등 다중 객체가 정상적으로 추적되었으며, 각 객체의 클래스명(car, truck 등)과 이동 궤적이 시각화됩니다.
- SORT의 알고리즘적 한계로 인해, 객체가 화면 밖으로 나갔다가 다시 들어오면 새로운 ID가 부여되는 **ID Switch** 현상이 관찰됩니다. 이는 SORT에 재식별(Re-ID) 신경망이 없기 때문이며, Deep SORT 같은 확장 알고리즘으로 해결할 수 있습니다.

---

## 실습 02 — Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화

### 1. 과제에 대한 설명
Mediapipe의 FaceMesh 모듈을 사용하여 얼굴의 468개 랜드마크를 추출하고, 이를 실시간 웹캠 영상에 시각화하는 프로그램을 구현합니다.
- Mediapipe의 `solutions.face_mesh`를 사용하여 얼굴 랜드마크 검출기를 초기화합니다.
- OpenCV를 사용하여 웹캠으로부터 실시간 영상을 캡처합니다.
- 검출된 얼굴 랜드마크를 실시간 영상에 점(`cv.circle`)으로 표시합니다.
- 랜드마크 좌표는 정규화(0~1)되어 있으므로, 이미지 크기에 맞게 변환합니다.
- **주의**: OpenCV는 BGR, MediaPipe는 RGB를 요구하므로 `cv.cvtColor(frame, cv.COLOR_BGR2RGB)` 변환이 필수입니다.
- 실행 시작부터 ESC 종료까지 전체 영상이 `result/02_facemesh_landmark.mp4`로 저장됩니다.
- ESC 키를 누르면 프로그램이 종료됩니다.

### 2. 핵심 코드 설명
```python
# [핵심] FaceMesh 검출기 초기화 - 468개 3D 랜드마크를 추출하는 모듈
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,       # 비디오 모드 (이전 프레임 추적 활용, 고속)
    max_num_faces=2,               # 최대 2명까지 검출
    refine_landmarks=True,         # 눈동자(Irises) 랜드마크 포함
    min_detection_confidence=0.5,  # 최소 검출 신뢰도
    min_tracking_confidence=0.5,   # 최소 추적 신뢰도
)

# [핵심] BGR -> RGB 변환 - MediaPipe는 RGB 입력 필수!
rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

# [핵심] FaceMesh 추론 실행
results = face_mesh.process(rgb_frame)

# [핵심] 정규화 좌표를 픽셀 좌표로 변환 후 점으로 시각화
for lm in face_landmarks.landmark:
    px = int(lm.x * w)  # 정규화 x좌표 -> 픽셀 좌표
    py = int(lm.y * h)  # 정규화 y좌표 -> 픽셀 좌표
    cv.circle(frame, (px, py), 1, (0, 255, 0), -1)  # 초록색 점 표시

# [핵심] 결과 비디오 저장 - 실행 시작부터 ESC 종료까지 전체 녹화
fourcc = cv.VideoWriter_fourcc(*"mp4v")
out = cv.VideoWriter(OUTPUT_VIDEO, fourcc, cam_fps, (width, height))
out.write(frame)  # 매 프레임마다 비디오에 기록
```

### 3. 전체 코드

```python
import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import time

# 현재 스크립트가 위치한 디렉터리를 기준으로 경로를 설정한다
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 결과 파일을 저장할 폴더 경로를 설정한다
RESULT_DIR = os.path.join(BASE_DIR, "result")
# 폴더가 없으면 생성하고, 이미 존재하면 에러 없이 넘어간다
os.makedirs(RESULT_DIR, exist_ok=True)

# 결과 비디오 저장 경로 (실행 시작부터 ESC 종료까지 전체 녹화)
OUTPUT_VIDEO = os.path.join(RESULT_DIR, "02_facemesh_landmark.mp4")



def main():
    """
    메인 실행 함수

    전체 파이프라인:
      1. MediaPipe FaceMesh 검출기 초기화
      2. 웹캠으로부터 실시간 영상 캡처
      3. 결과 비디오 저장용 VideoWriter 설정
      4. 프레임 루프:
         a. BGR -> RGB 변환 (MediaPipe 입력 요구사항)
         b. FaceMesh 추론으로 468개 랜드마크 추출
         c. 랜드마크를 점(circle)으로 시각화
         d. 그물망(Tesselation), 윤곽선(Contours), 눈동자(Irises) 시각화
         e. 결과 프레임을 비디오 파일에 기록
      5. ESC 키로 종료 및 비디오 저장 완료
    """
    # 구분선을 출력하여 출력 결과의 가독성을 높인다
    print("=" * 60)
    # 과제 제목을 출력한다
    print("과제 02: MediaPipe FaceMesh 얼굴 랜드마크 시각화")
    # 구분선을 출력한다
    print("=" * 60)

    # ── 1. MediaPipe FaceMesh 초기화 ──
    # mp.solutions.face_mesh를 사용하여 얼굴 랜드마크 검출기를 생성한다
    mp_face_mesh = mp.solutions.face_mesh

    # FaceMesh 객체를 생성하고 파라미터를 설정한다
    face_mesh = mp_face_mesh.FaceMesh(
        # False = 비디오 모드: 이전 프레임의 모션 정보를 활용하여 추적 (고속)
        # True로 설정하면 매 프레임 독립적으로 검출 (정적 이미지 분석용)
        static_image_mode=False,
        # 동시에 검출할 최대 얼굴 수를 2로 설정한다
        max_num_faces=2,
        # True로 설정하면 눈동자(Irises) 주변 랜드마크를 추가로 정밀 검출한다
        refine_landmarks=True,
        # 얼굴 검출의 최소 신뢰도를 설정한다 (BlazeFace가 사용)
        min_detection_confidence=0.5,
        # 랜드마크 추적의 최소 신뢰도를 설정한다 (이 값 이하로 떨어지면 BlazeFace 재검출)
        min_tracking_confidence=0.5,
    )

    # MediaPipe 그리기 유틸리티를 가져온다 (그물망, 윤곽선 등 시각화용)
    mp_drawing = mp.solutions.drawing_utils
    # MediaPipe 기본 그리기 스타일을 가져온다 (색상, 두께 등 미리 정의된 스타일)
    mp_drawing_styles = mp.solutions.drawing_styles

    # 초기화 완료 메시지를 출력한다
    print("[INFO] FaceMesh 검출기 초기화 완료 (468개 3D 랜드마크)")

    # ── 2. 웹캠 열기 ──
    # OpenCV를 사용하여 웹캠(장치 인덱스 0)으로부터 실시간 영상을 캡처한다
    cap = cv.VideoCapture(0)
    # 웹캠을 열 수 없으면 에러 메시지를 출력하고 종료한다
    if not cap.isOpened():
        # 웹캠 열기 실패 에러를 출력한다
        print("[ERROR] 웹캠을 열 수 없습니다.")
        # 해결 방법 안내를 출력한다
        print("        웹캠이 연결되어 있는지 확인하세요.")
        # 함수를 조기 종료한다
        return

    # 웹캠의 가로 해상도를 정수로 가져온다
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # 웹캠의 세로 해상도를 정수로 가져온다
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # 웹캠의 FPS(초당 프레임 수)를 가져온다 (비디오 저장 시 사용)
    cam_fps = cap.get(cv.CAP_PROP_FPS)
    # FPS가 유효하지 않으면 기본값 30으로 설정한다
    if cam_fps <= 0:
        cam_fps = 30.0
    # 웹캠 해상도 정보를 출력한다
    print(f"[INFO] 웹캠 해상도: {width}x{height}, {cam_fps:.0f}FPS")
    # 종료 방법을 안내한다
    print("[INFO] ESC 키를 누르면 프로그램이 종료되고 비디오가 저장됩니다.")
    # 구분선을 출력한다
    print("-" * 60)

    # ── 3. 결과 비디오 저장용 VideoWriter 설정 ──
    # mp4v 코덱을 4자리 문자로 설정한다
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # VideoWriter 객체를 생성한다 (출력경로, 코덱, FPS, 해상도)
    out = cv.VideoWriter(OUTPUT_VIDEO, fourcc, cam_fps, (width, height))
    # 비디오 녹화 시작 메시지를 출력한다
    print(f"[INFO] 비디오 녹화 시작: {OUTPUT_VIDEO}")

    # 프레임 카운터를 0으로 초기화한다
    frame_count = 0

    # FPS 계산을 위한 이전 시간을 현재 시간으로 기록한다
    prev_time = time.time()

    # ── 4. 프레임 처리 루프 ──
    # 무한 루프로 매 프레임을 처리한다 (ESC 키 입력 시 탈출)
    while True:
        # OpenCV로 웹캠에서 한 프레임을 읽어온다
        ret, frame = cap.read()
        # 프레임을 읽을 수 없으면 에러를 출력하고 루프를 종료한다
        if not ret:
            # 프레임 읽기 실패 에러를 출력한다
            print("[ERROR] 프레임을 읽을 수 없습니다.")
            # 루프를 종료한다
            break

        # 프레임 카운터를 1 증가시킨다
        frame_count += 1

        # ────────────────────────────────────
        # 핵심: BGR -> RGB 변환
        # ────────────────────────────────────
        # OpenCV의 기본 컬러 포맷은 BGR이지만,
        # MediaPipe는 RGB를 기대하므로 반드시 변환해야 한다.
        # 이 변환이 빠지면 검출 성능이 급락하거나 아예 동작하지 않는다.
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # 성능 최적화: 이미지를 읽기 전용으로 설정하면
        # MediaPipe 내부에서 불필요한 복사를 피할 수 있다
        rgb_frame.flags.writeable = False

        # ────────────────────────────────────
        # FaceMesh 추론 실행
        # ────────────────────────────────────
        # process()를 호출하면 내부적으로:
        #   - static_image_mode=False이므로 이전 프레임의 추적 정보를 활용
        #   - 신뢰도가 떨어지면 자동으로 BlazeFace 재검출
        results = face_mesh.process(rgb_frame)

        # 이미지를 다시 쓰기 가능으로 복원한다 (시각화를 위해)
        rgb_frame.flags.writeable = True

        # ────────────────────────────────────
        # 랜드마크 시각화
        # ────────────────────────────────────
        # 검출된 얼굴 수를 0으로 초기화한다
        num_faces = 0

        # 얼굴 랜드마크가 검출되었는지 확인한다
        if results.multi_face_landmarks:
            # 검출된 얼굴 수를 저장한다
            num_faces = len(results.multi_face_landmarks)

            # 검출된 각 얼굴의 랜드마크를 순회한다
            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # ── (a) 그물망(Tesselation) 그리기 ──
                # 468개 랜드마크를 삼각형으로 연결하여 얼굴 표면 구조를 표현한다
                mp_drawing.draw_landmarks(
                    # 그릴 대상 이미지를 지정한다
                    image=frame,
                    # 검출된 랜드마크 데이터를 전달한다
                    landmark_list=face_landmarks,
                    # 그물망 연결 관계를 지정한다
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    # 그물망 모드에서는 점 생략한다
                    landmark_drawing_spec=None,
                    # MediaPipe 기본 그물망 스타일을 적용한다
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

                # ── (b) 윤곽선(Contours) 그리기 ──
                # 얼굴 외곽, 눈, 눈썹, 입술, 코의 윤곽선을 강조한다
                mp_drawing.draw_landmarks(
                    # 그릴 대상 이미지를 지정한다
                    image=frame,
                    # 검출된 랜드마크 데이터를 전달한다
                    landmark_list=face_landmarks,
                    # 윤곽선 연결 관계를 지정한다
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    # 윤곽선 모드에서는 점 생략한다
                    landmark_drawing_spec=None,
                    # MediaPipe 기본 윤곽선 스타일을 적용한다
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

                # ── (c) 눈동자(Irises) 그리기 ──
                # refine_landmarks=True일 때 눈동자 주변 랜드마크를 시각화한다
                mp_drawing.draw_landmarks(
                    # 그릴 대상 이미지를 지정한다
                    image=frame,
                    # 검출된 랜드마크 데이터를 전달한다
                    landmark_list=face_landmarks,
                    # 눈동자 연결 관계를 지정한다
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    # 눈동자 모드에서는 점 생략한다
                    landmark_drawing_spec=None,
                    # MediaPipe 기본 눈동자 스타일을 적용한다
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

                # ── (d) 468개 랜드마크를 개별 점(circle)으로 표시 ──
                # 과제 요구사항: "검출된 얼굴 랜드마크를 실시간 영상에 점으로 표시합니다"
                # 프레임의 높이(h)와 너비(w)를 가져온다
                h, w = frame.shape[:2]
                # 468개 랜드마크를 인덱스와 함께 순회한다
                for idx, lm in enumerate(face_landmarks.landmark):
                    # 정규화 x좌표(0~1)를 픽셀 좌표(정수)로 변환한다
                    px = int(lm.x * w)
                    # 정규화 y좌표(0~1)를 픽셀 좌표(정수)로 변환한다
                    py = int(lm.y * h)
                    # OpenCV의 circle 함수로 각 랜드마크를 반지름 1의 초록색 점으로 표시한다
                    cv.circle(frame, (px, py), 1, (0, 255, 0), -1)

        # 현재 시간을 가져와 FPS를 계산한다
        curr_time = time.time()
        # 이전 프레임과의 시간 차이로 FPS 값을 계산한다
        fps_val = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        # 현재 시간을 이전 시간으로 갱신한다
        prev_time = curr_time

        # 화면 상단에 표시할 정보 텍스트를 구성한다
        info_text = f"Faces: {num_faces} | Landmarks: {num_faces * 468} | FPS: {fps_val:.1f} | ESC to quit"
        # 텍스트 배경을 검은색 사각형으로 깔아서 가독성을 높인다
        cv.rectangle(frame, (0, 0), (520, 35), (0, 0, 0), -1)
        # 정보 텍스트를 초록색으로 화면에 출력한다
        cv.putText(
            frame, info_text, (10, 25),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )


        # 현재 프레임을 결과 비디오 파일에 기록한다
        out.write(frame)

        # 실시간으로 화면에 프레임을 출력한다
        cv.imshow("FaceMesh Landmark Detection", frame)

        # 1ms 동안 키 입력을 대기하고, 하위 8비트만 추출한다
        key = cv.waitKey(1) & 0xFF
        # ESC 키(코드 27)가 눌리면 프로그램을 종료한다
        if key == 27:
            # 종료 메시지를 출력한다
            print("[INFO] ESC 키 입력 - 프로그램 종료")
            # 루프를 탈출한다
            break

    # ── 5. 자원 해제 ──
    # 웹캠 장치를 해제한다
    cap.release()
    # 비디오 파일 쓰기를 완료하고 파일을 닫는다
    out.release()
    # 모든 OpenCV 창을 닫는다
    cv.destroyAllWindows()
    # FaceMesh 검출기를 해제한다
    face_mesh.close()

    # 처리 완료 메시지를 출력한다
    print(f"\n[완료] 총 {frame_count}프레임 처리")
    # 결과 비디오 저장 경로를 출력한다
    print(f"[저장] 결과 비디오: {OUTPUT_VIDEO}")
    # 구분선을 출력한다
    print("=" * 60)


# 이 파일이 직접 실행될 때만 main 함수를 호출한다
if __name__ == "__main__":
    # main 함수를 실행한다
    main()
```

### 4. 최종 결과물
*(FaceMesh 랜드마크 검출 결과 - 웹캠에서 실시간으로 468개 랜드마크가 점으로 표시되며, 그물망/윤곽선/눈동자가 함께 시각화됩니다.)*

결과 비디오: 

https://github.com/user-attachments/assets/e47c0efb-0c29-453f-a371-63fe9f13922d

(실행 시작~ESC 종료까지 전체 녹화)

### 5. 결과 해석
- MediaPipe FaceMesh를 사용하여 **468개의 3D 얼굴 랜드마크**를 실시간으로 검출하고 시각화했습니다.
- `static_image_mode=False`로 설정하여 비디오 모드로 동작시켰으며, 이전 프레임의 추적 정보를 활용하여 매 프레임 얼굴 검출을 생략함으로써 높은 FPS를 달성했습니다.
- BGR->RGB 변환(`cv.cvtColor`)을 올바르게 적용하여 MediaPipe의 정확한 검출 성능을 보장했습니다.
- 랜드마크 좌표가 정규화(0~1)되어 있으므로 `int(lm.x * w)`, `int(lm.y * h)`로 픽셀 좌표 변환 후 `cv.circle()`로 시각화했습니다.
- 그물망(Tesselation), 윤곽선(Contours), 눈동자(Irises) 시각화를 함께 적용하여 풍부한 얼굴 구조 정보를 확인할 수 있습니다.
- 실행 전체 영상이 MP4로 저장되어 과제 결과물로 제출할 수 있습니다.

---

## 정리

이번 실습은 강의에서 다룬 비디오 분석(추적, MediaPipe 사람 인식)의 개념을 직접 구현한 과제입니다.
- 1번에서는 YOLOv3 검출기와 SORT 알고리즘(칼만 필터 + 헝가리안 알고리즘)을 결합하여 비디오에서 다중 객체를 추적하고, 각 객체에 고유 ID를 부여하여 시각화했습니다.
- 2번에서는 MediaPipe의 FaceMesh 모듈로 468개 얼굴 랜드마크를 실시간 추출하고, OpenCV의 circle 함수로 점 시각화를 구현했습니다.
- 두 과제 모두 과제 요구사항(객체 검출, 추적기 초기화, 추적 유지, ID 시각화 / FaceMesh 초기화, 웹캠 캡처, 점 표시, ESC 종료)을 완전히 충족합니다.

모든 결과물은 `result/` 폴더에 자동으로 저장됩니다.
