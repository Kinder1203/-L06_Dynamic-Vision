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
