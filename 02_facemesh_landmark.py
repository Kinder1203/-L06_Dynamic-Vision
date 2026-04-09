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
