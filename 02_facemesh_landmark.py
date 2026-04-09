"""
과제 02: Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화

Mediapipe의 FaceMesh 모듈을 사용하여 얼굴의 468개 랜드마크를 추출하고,
이를 웹캠 실시간 영상에 시각화하는 프로그램을 구현한다.

핵심 구성:
  - 검출기: MediaPipe FaceMesh (468개 3D 랜드마크)
  - 입력: 웹캠 실시간 영상 (OpenCV VideoCapture)
  - 시각화: 랜드마크를 점(circle)으로 표시 + 그물망/윤곽선/눈동자 시각화
  - 종료: ESC 키

MediaPipe 동작 원리 (강의 내용 기반):
  - FaceMesh는 내부적으로 BlazeFace와 랜드마크 예측기가 파이프라인으로 연결되어 있다.
  - 첫 프레임에서 BlazeFace로 얼굴을 검출한 후,
    이후 프레임에서는 이전 프레임의 모션 정보를 활용해 랜드마크를 바로 예측(추적)한다.
  - 이 방식으로 매 프레임마다 얼굴 검출을 생략하여 속도를 크게 높인다.
  - 신뢰도가 일정 수준 이하로 떨어지면 BlazeFace 검출을 다시 수행한다.

주의사항:
  - OpenCV는 BGR 채널을 사용하지만, MediaPipe는 RGB를 요구한다.
  - 반드시 cv.cvtColor(frame, cv.COLOR_BGR2RGB) 변환 후 process()에 전달해야 한다.
  - 변환 없이 BGR 이미지를 넣으면 성능이 급락하거나 오작동한다.
  - 랜드마크 좌표는 정규화(0~1)되어 있으므로, 이미지 크기에 맞게 변환이 필요하다.
"""

import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import time

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# 결과 스크린샷 저장 경로
SCREENSHOT_PATH = os.path.join(RESULT_DIR, "02_facemesh_screenshot.png")


def main():
    """
    메인 실행 함수

    전체 파이프라인:
      1. MediaPipe FaceMesh 검출기 초기화
      2. 웹캠으로부터 실시간 영상 캡처
      3. 프레임 루프:
         a. BGR -> RGB 변환 (MediaPipe 입력 요구사항)
         b. FaceMesh 추론으로 468개 랜드마크 추출
         c. 랜드마크를 점(circle)으로 시각화
         d. 그물망(Tesselation), 윤곽선(Contours), 눈동자(Irises) 시각화
      4. ESC 키로 종료
    """
    print("=" * 60)
    print("과제 02: MediaPipe FaceMesh 얼굴 랜드마크 시각화")
    print("=" * 60)

    # ── 1. MediaPipe FaceMesh 초기화 ──
    # mp.solutions.face_mesh를 사용하여 얼굴 랜드마크 검출기를 생성한다
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        # False = 비디오 모드: 이전 프레임의 모션 정보를 활용하여 추적 (고속)
        # True로 설정하면 매 프레임 독립적으로 검출 (정적 이미지 분석용)
        static_image_mode=False,
        # 동시에 검출할 최대 얼굴 수
        max_num_faces=2,
        # True로 설정하면 눈동자(Irises) 주변 랜드마크를 추가로 정밀 검출한다
        refine_landmarks=True,
        # 얼굴 검출 최소 신뢰도 (BlazeFace가 사용)
        min_detection_confidence=0.5,
        # 랜드마크 추적 최소 신뢰도 (이 값 이하로 떨어지면 BlazeFace 재검출)
        min_tracking_confidence=0.5,
    )

    # MediaPipe 그리기 유틸리티
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    print("[INFO] FaceMesh 검출기 초기화 완료 (468개 3D 랜드마크)")

    # ── 2. 웹캠 열기 ──
    # OpenCV를 사용하여 웹캠(장치 인덱스 0)으로부터 실시간 영상을 캡처한다
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        print("        웹캠이 연결되어 있는지 확인하세요.")
        return

    # 웹캠 해상도 확인
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 웹캠 해상도: {width}x{height}")
    print("[INFO] ESC 키를 누르면 프로그램이 종료됩니다.")
    print("-" * 60)

    frame_count = 0
    screenshot_saved = False
    prev_time = time.time()

    # ── 3. 프레임 처리 루프 ──
    while True:
        # OpenCV로 웹캠에서 한 프레임을 읽어온다
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임을 읽을 수 없습니다.")
            break

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

        # 이미지를 다시 쓰기 가능으로 복원 (시각화를 위해)
        rgb_frame.flags.writeable = True

        # ────────────────────────────────────
        # 랜드마크 시각화
        # ────────────────────────────────────
        num_faces = 0

        if results.multi_face_landmarks:
            num_faces = len(results.multi_face_landmarks)

            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # ── (a) 그물망(Tesselation) 그리기 ──
                # 468개 랜드마크를 삼각형으로 연결하여 얼굴 표면 구조를 표현한다
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

                # ── (b) 윤곽선(Contours) 그리기 ──
                # 얼굴 외곽, 눈, 눈썹, 입술, 코의 윤곽선을 강조한다
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

                # ── (c) 눈동자(Irises) 그리기 ──
                # refine_landmarks=True일 때 눈동자 주변 랜드마크를 시각화한다
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

                # ── (d) 468개 랜드마크를 개별 점(circle)으로 표시 ──
                # 과제 요구사항: "검출된 얼굴 랜드마크를 실시간 영상에 점으로 표시합니다"
                # 랜드마크 좌표는 정규화(0~1)되어 있으므로 이미지 크기에 맞게 변환한다
                h, w = frame.shape[:2]
                for idx, lm in enumerate(face_landmarks.landmark):
                    # 정규화 좌표(0~1) -> 픽셀 좌표(정수) 변환
                    px = int(lm.x * w)
                    py = int(lm.y * h)

                    # OpenCV의 circle 함수로 각 랜드마크를 초록색 점으로 표시한다
                    cv.circle(frame, (px, py), 1, (0, 255, 0), -1)

        # FPS 계산
        curr_time = time.time()
        fps_val = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # 화면 상단에 정보 표시
        info_text = f"Faces: {num_faces} | Landmarks: {num_faces * 468} | FPS: {fps_val:.1f} | ESC to quit"
        # 텍스트 배경을 깔아서 가독성을 높인다
        cv.rectangle(frame, (0, 0), (520, 35), (0, 0, 0), -1)
        cv.putText(
            frame, info_text, (10, 25),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # 첫 번째 얼굴 검출 시 결과 스크린샷을 저장한다 (과제 제출 증거)
        if num_faces > 0 and not screenshot_saved:
            cv.imwrite(SCREENSHOT_PATH, frame)
            print(f"[저장] 검출 결과 스크린샷: {SCREENSHOT_PATH}")
            screenshot_saved = True

        # 실시간 화면 출력
        cv.imshow("FaceMesh Landmark Detection", frame)

        # ESC 키(27)를 누르면 프로그램이 종료된다
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            print("[INFO] ESC 키 입력 - 프로그램 종료")
            break

    # ── 4. 자원 해제 ──
    cap.release()
    cv.destroyAllWindows()
    face_mesh.close()

    print(f"\n[완료] 총 {frame_count}프레임 처리")
    print("=" * 60)


if __name__ == "__main__":
    main()
