"""
SORT: A Simple, Online and Realtime Tracker
Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

이 모듈은 칼만 필터(Kalman Filter)와 헝가리안 알고리즘(Hungarian Algorithm)을
사용하여 다중 객체를 추적하는 SORT 알고리즘을 구현합니다.

참고: https://github.com/abewley/sort
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    """
    헝가리안 알고리즘을 사용하여 비용 행렬(cost matrix)에서
    전체 비용이 최소가 되는 최적의 1:1 매칭을 수행합니다.
    """
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    두 바운딩 박스 집합 간의 IoU(Intersection over Union)를 일괄 계산합니다.
    입력 형식: [x1, y1, x2, y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    # 교집합 영역 계산
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h

    # IoU = 교집합 / 합집합
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


def convert_bbox_to_z(bbox):
    """
    바운딩 박스 [x1, y1, x2, y2]를 칼만 필터 상태 벡터 [x, y, s, r]로 변환합니다.
    x, y: 중심 좌표 / s: 면적(scale) / r: 종횡비(aspect ratio)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    칼만 필터 상태 벡터 [x, y, s, r]을 바운딩 박스 [x1, y1, x2, y2]로 역변환합니다.
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


class KalmanBoxTracker:
    """
    개별 추적 대상의 내부 상태를 칼만 필터로 관리하는 클래스입니다.
    상태 벡터: [x, y, s, r, dx, dy, ds] — 위치, 면적, 종횡비 + 각각의 속도
    """

    count = 0

    def __init__(self, bbox):
        """초기 바운딩 박스로 칼만 필터 트래커를 생성합니다."""
        # 등속 운동 모델(constant velocity model) 정의
        # 상태 차원 7 (x, y, s, r, dx, dy, ds), 관측 차원 4 (x, y, s, r)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # 상태 전이 행렬 F: 현재 상태에서 다음 상태를 예측
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        # 관측 행렬 H: 상태 벡터에서 관측 가능한 부분만 추출
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # 관측 노이즈, 초기 불확실성, 프로세스 노이즈 설정
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # 초기 속도에 대한 높은 불확실성
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # 초기 바운딩 박스로 상태 벡터 초기화
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """관측된 바운딩 박스로 칼만 필터 상태를 보정(update)합니다."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """칼만 필터로 다음 프레임의 위치를 예측합니다."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """현재 바운딩 박스 추정값을 반환합니다."""
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    검출된 객체와 기존 트래커를 IoU 기반으로 매칭합니다.
    반환: (매칭된 쌍, 미매칭 검출, 미매칭 트래커)
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    # IoU 비용 행렬 계산
    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # 헝가리안 알고리즘으로 최적 매칭 수행 (비용 = -IoU)
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    # 매칭되지 않은 검출과 트래커 분류
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # IoU가 임계값 미만인 매칭은 제외
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

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
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        검출 결과를 받아 추적을 수행합니다.

        입력: dets — [[x1, y1, x2, y2, score], ...] 형태의 numpy 배열
        출력: [[x1, y1, x2, y2, track_id], ...] 형태의 numpy 배열

        주의: 검출이 없는 프레임에서도 반드시 빈 배열로 호출해야 합니다.
        """
        self.frame_count += 1

        # 기존 트래커들의 예측 위치 계산
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 검출-트래커 매칭 수행
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # 매칭된 트래커의 상태를 검출 결과로 업데이트
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 미매칭 검출에 대해 새 트래커 생성
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # 결과 수집 및 죽은 트랙 제거
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits
                or self.frame_count <= self.min_hits
            ):
                # ID는 +1 (MOT 벤치마크 규약: 양수 ID)
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # max_age 초과 시 트랙 삭제
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
