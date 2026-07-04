import cv2
import numpy as np
import time
from collections import deque

# =====================================================
# FPS(Frame Per Second) 계산을 위한 변수
# 최근 30프레임의 FPS를 저장하여 평균 FPS를 계산한다.
# =====================================================
fps_history = deque(maxlen=30)

# 이전 프레임이 처리된 시간을 저장
previous_time = time.time()


# =====================================================
# 컬러(BGR) 영상을 흑백(Grayscale) 영상으로 변환하는 함수
# 입력 : 컬러 프레임(frame)
# 출력 : 흑백 이미지(gray)
# =====================================================
def to_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


# =====================================================
# Gaussian Blur를 적용하여 노이즈를 제거하는 함수
# 입력 : 흑백 이미지(gray)
# 출력 : Blur가 적용된 이미지(blur)
# =====================================================
def apply_blur(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur


# =====================================================
# Canny Edge Detection을 수행하는 함수
# 입력 : Blur 이미지
# 출력 : 윤곽선(Edge) 이미지
# =====================================================
def detect_edges(blur):
    edges = cv2.Canny(blur, 100, 200)
    return edges


# =====================================================
# 전체 전처리 파이프라인
#
# 컬러 영상
#      ↓
#  Grayscale
#      ↓
# Gaussian Blur
#      ↓
# Canny Edge
#
# 세 결과를 모두 반환한다.
# =====================================================
def preprocess(frame):
    gray = to_grayscale(frame)
    blur = apply_blur(gray)
    edges = detect_edges(blur)

    return gray, blur, edges


# =====================================================
# FPS(Frame Per Second)를 계산하는 함수
#
# 현재 시간과 이전 시간의 차이를 이용하여
# 순간 FPS를 계산한 뒤,
# 최근 30프레임 평균 FPS를 반환한다.
# =====================================================
def calculate_fps():
    global previous_time

    # 현재 시간 저장
    current_time = time.time()

    # 프레임 처리 시간 계산
    elapsed_time = current_time - previous_time

    # 0으로 나누는 오류 방지
    if elapsed_time <= 0:
        return 0.0

    # 순간 FPS 계산
    fps = 1 / elapsed_time

    # 현재 시간을 다음 계산을 위한 이전 시간으로 저장
    previous_time = current_time

    # 최근 FPS 저장
    fps_history.append(fps)

    # 평균 FPS 계산
    average_fps = sum(fps_history) / len(fps_history)

    return average_fps


# =====================================================
# 화면 좌측 상단에 FPS를 출력하는 함수
#
# 입력 : frame, fps
# 출력 : FPS가 표시된 frame
# =====================================================
def draw_fps(frame, fps):

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",          # 출력 문자열
        (10, 30),                  # 출력 위치
        cv2.FONT_HERSHEY_SIMPLEX,  # 폰트
        1,                         # 글자 크기
        (0, 255, 0),               # 초록색(BGR)
        2                          # 글자 두께
    )

    return frame


# =====================================================
# 원본 영상과 Edge 영상을 좌우로 합치는 함수
#
# Edge는 흑백(1채널)이므로
# BGR(3채널)로 변환한 후 합친다.
# =====================================================
def combine_frames(frame, edges):

    # Edge를 컬러(BGR) 이미지로 변환
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 좌우로 영상 합치기
    combined = np.hstack((frame, edges_color))

    return combined


# =====================================================
# 웹캠 연결
# =====================================================
cap = cv2.VideoCapture(0)

# 웹캠 연결 여부 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠 연결 성공!")

# 웹캠 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# =====================================================
# 메인 루프
# 계속해서 프레임을 읽고 처리한다.
# =====================================================
while True:

    # 웹캠으로부터 프레임 읽기
    ret, frame = cap.read()

    # 프레임 읽기에 실패하면 종료
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 셀카처럼 보이도록 좌우 반전
    frame = cv2.flip(frame, 1)

    # 전처리 수행
    gray, blur, edges = preprocess(frame)

    # FPS 계산
    fps = calculate_fps()

    # FPS를 원본 영상에 출력
    frame_with_fps = draw_fps(frame.copy(), fps)

    # 원본 영상과 Edge 영상을 좌우로 합침
    combined = combine_frames(frame_with_fps, edges)

    # 결과 출력
    cv2.imshow("Original + Edges", combined)

    # q 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =====================================================
# 프로그램 종료
# 웹캠 자원 해제 및 창 닫기
# =====================================================
cap.release()
cv2.destroyAllWindows()