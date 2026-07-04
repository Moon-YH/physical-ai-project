import cv2
import numpy as np
import time
from collections import deque

fps_history = deque(maxlen=30)
previous_time = time.time()


def to_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def apply_blur(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur


def detect_edges(blur):
    edges = cv2.Canny(blur, 100, 200)
    return edges


def preprocess(frame):
    gray = to_grayscale(frame)
    blur = apply_blur(gray)
    edges = detect_edges(blur)

    return gray, blur, edges


def calculate_fps():
    global previous_time

    current_time = time.time()
    elapsed_time = current_time - previous_time

    if elapsed_time <= 0:
        return 0.0

    fps = 1 / elapsed_time
    previous_time = current_time

    fps_history.append(fps)
    average_fps = sum(fps_history) / len(fps_history)

    return average_fps


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return frame


def combine_frames(frame, edges):
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame, edges_color))

    return combined


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠 연결 완료")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    frame = cv2.flip(frame, 1)

    gray, blur, edges = preprocess(frame)

    fps = calculate_fps()

    frame_with_fps = draw_fps(frame.copy(), fps)

    combined = combine_frames(frame_with_fps, edges)

    cv2.imshow("Original + Edges", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()