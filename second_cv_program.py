from __future__ import print_function
import cv2 as cv
import argparse
from typing import Tuple, List, Optional, Union
import time
import os
import sys

import numpy as np

# Глобальные переменные для статистики
total_faces: int = 0
total_eyes: int = 0
frame_count: int = 0
start_time: Optional[float] = None
fps_list: List[float] = []

def get_video_info(cap: cv.VideoCapture) -> Tuple[int, int, float, int]:
    """Получает и выводит информацию о видео.

    Args:
        cap: Объект видеозахвата OpenCV.

    Returns:
        Кортеж с шириной, высотой, FPS и количеством кадров.
    """
    width: int = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps: float = cap.get(cv.CAP_PROP_FPS)
    frame_count: int = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    codec: int = int(cap.get(cv.CAP_PROP_FOURCC))
    codec_str: str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

    print("\n" + "=" * 60)
    print("VIDEO INFORMATION")
    print("=" * 60)
    print(f"Resolution:      {width}x{height}")
    print(f"FPS:             {fps:.2f}")
    print(f"Total frames:    {frame_count}")
    print(f"Duration:        {frame_count / fps:.2f} seconds" if fps > 0 else "Duration:        N/A")
    print(f"Codec:           {codec_str}")
    print("=" * 60 + "\n")

    return width, height, fps, frame_count

def detectAndDisplay(frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Обнаруживает лица и глаза, возвращает статистику.

    Args:
        frame: Кадр изображения в формате NumPy массива (BGR).

    Returns:
        Кортеж из обработанного кадра, количества лиц и глаз в кадре.
    """
    global total_faces, total_eyes, frame_count

    frame_gray: np.ndarray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    faces_in_frame: int = len(faces)
    eyes_in_frame: int = 0

    for face in faces:
        x, y, w, h = face
        center: Tuple[int, int] = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)

        cv.putText(frame, f"Face {faces_in_frame}", (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        faceROI: np.ndarray = frame_gray[y:y + h, x:x + w]

        eyes: np.ndarray = eyes_cascade.detectMultiScale(faceROI)
        for eye in eyes:
            x2, y2, w2, h2 = eye
            eye_center: Tuple[int, int] = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius: int = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
            eyes_in_frame += 1

    total_faces += faces_in_frame
    total_eyes += eyes_in_frame
    frame_count += 1

    return frame, faces_in_frame, eyes_in_frame

def draw_stats(frame: np.ndarray,
               faces_count: int,
               eyes_count: int,
               current_fps: float) -> np.ndarray:
    """Рисует статистику на кадре.

    Args:
        frame: Кадр изображения в формате NumPy массива (BGR).
        faces_count: Количество лиц в текущем кадре.
        eyes_count: Количество глаз в текущем кадре.
        current_fps: Текущая частота кадров в секунду.

    Returns:
        Обработанный кадр с наложенной статистикой.
    """
    height, width = frame.shape[:2]

    overlay: np.ndarray = frame.copy()
    cv.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    stats: List[str] = [
        f"Frame: {frame_count}",
        f"FPS: {current_fps:.1f}",
        f"Faces detected: {faces_count}",
        f"Eyes detected: {eyes_count}",
        f"Total faces: {total_faces}",
    ]

    y_offset: int = 35
    for i, stat in enumerate(stats):
        cv.putText(frame, stat, (20, y_offset + i * 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

def save_screenshot(frame: np.ndarray, faces_count: int) -> None:
    """Сохраняет скриншот при обнаружении лиц.

    Args:
        frame: Кадр изображения в формате NumPy массива (BGR).
        faces_count: Количество лиц в текущем кадре.
    """
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')

    timestamp: str = time.strftime("%Y%m%d_%H%M%S")
    filename: str = f"screenshots/face_detection_{timestamp}_faces{faces_count}.jpg"
    cv.imwrite(filename, frame)
    print(f"\n📸 Screenshot saved: {filename}")

def print_final_stats(total_frames: int, duration: float) -> None:
    """Выводит финальную статистику.

    Args:
        total_frames: Общее количество обработанных кадров.
        duration: Длительность обработки в секундах.
    """
    print("\n" + "=" * 60)
    print("DETECTION STATISTICS")
    print("=" * 60)
    print(f"Total frames processed:    {total_frames}")
    print(f"Total faces detected:      {total_faces}")
    print(f"Total eyes detected:       {total_eyes}")
    print(
        f"Average faces per frame:   {total_faces / total_frames:.2f}" if total_frames > 0 else "Average faces per frame:   0")
    print(
        f"Average eyes per frame:    {total_eyes / total_frames:.2f}" if total_frames > 0 else "Average eyes per frame:    0")
    print(f"Processing duration:       {duration:.2f} seconds")
    print(
        f"Average FPS:               {sum(fps_list) / len(fps_list):.2f}" if fps_list else "Average FPS:               N/A")
    print("=" * 60)
    print("\nControls:")
    print("  Q     - Quit/Exit")
    print("  S     - Save screenshot")
    print("  P     - Pause/Resume")
    print("  SPACE - Next frame (when paused)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    print("\n🎥 Checking available cameras...")
    available_cameras: List[int] = []
    for i in range(10):
        cap: cv.VideoCapture = cv.VideoCapture(i)
        if cap.isOpened():
            ret: bool
            frame: Optional[np.ndarray]
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✓ Camera {i}: Working (resolution: {frame.shape})")
                available_cameras.append(i)
            else:
                print(f"⚠ Camera {i}: Opened but can't read frames")
            cap.release()

    if available_cameras:
        print(f"\n✓ Found {len(available_cameras)} working camera(s): {available_cameras}")
    else:
        print("\n⚠ No working cameras found. Use video file instead.")

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Enhanced Face Detection with Video Analytics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      python script.py --source video.mp4
      python script.py --source 0 --save-screenshots
      python script.py --source video.mp4 --skip-frames 2
        """
    )
    parser.add_argument('--face_cascade',
                        help='Path to face cascade.',
                        default='cv_haarcascade_xmls/haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes_cascade',
                        help='Path to eyes cascade.',
                        default='cv_haarcascade_xmls/haarcascade_eye_tree_eyeglasses.xml')
    parser.add_argument('--source',
                        help='Video file or camera index',
                        default='videos/business195.mov')
    parser.add_argument('--save-screenshots',
                        action='store_true',
                        help='Auto-save screenshots when faces detected')
    parser.add_argument('--skip-frames',
                        type=int,
                        default=0,
                        help='Skip N frames between processing (for speed)')
    parser.add_argument('--output',
                        help='Save processed video to file')
    args: argparse.Namespace = parser.parse_args()

    face_cascade: cv.CascadeClassifier = cv.CascadeClassifier()
    eyes_cascade: cv.CascadeClassifier = cv.CascadeClassifier()

    if not face_cascade.load(cv.samples.findFile(args.face_cascade)):
        print('❌ Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv.samples.findFile(args.eyes_cascade)):
        print('❌ Error loading eyes cascade')
        exit(0)

    print("✓ Cascades loaded successfully\n")

    # Открытие источника видео
    source: Union[int, str]
    try:
        source = int(args.source)
        print(f"📹 Opening camera {source}...")
    except ValueError:
        source = args.source
        print(f"📁 Opening video file: {source}...")

    cap: cv.VideoCapture = cv.VideoCapture(source)

    if not cap.isOpened():
        print(f'❌ Error: Could not open {source}')
        exit(0)

    print(f"✓ Successfully opened: {source}")

    # Получение информации о видео
    width: int
    height: int
    video_fps: float
    total_video_frames: int
    width, height, video_fps, total_video_frames = get_video_info(cap)

    # Настройка записи видео
    video_writer: Optional[cv.VideoWriter] = None
    if args.output:
        fourcc: int = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(args.output, fourcc, video_fps, (width, height))
        print(f"💾 Recording output to: {args.output}\n")

    start_time: float = time.time()
    frame_time: float = time.time()
    paused: bool = False
    skip_counter: int = 0

    print("🚀 Starting face detection... Press Q to exit\n")

    try:
        while True:
            if not paused:
                ret: bool
                frame: Optional[np.ndarray]
                ret, frame = cap.read()
                if not ret or frame is None:
                    print('\n✓ End of video reached')
                    break

                if args.skip_frames > 0:
                    skip_counter += 1
                    if skip_counter <= args.skip_frames:
                        continue
                    skip_counter = 0

                # Обработка кадра
                processed_frame: np.ndarray
                faces_count: int
                eyes_count: int
                processed_frame, faces_count, eyes_count = detectAndDisplay(frame)

                # Расчет FPS
                current_time: float = time.time()
                current_fps: float = 1 / (current_time - frame_time) if (current_time - frame_time) > 0 else 0
                frame_time = current_time
                fps_list.append(current_fps)

                display_frame: np.ndarray = draw_stats(processed_frame, faces_count, eyes_count, current_fps)

                if args.save_screenshots and faces_count > 0 and frame_count % 30 == 0:
                    save_screenshot(display_frame, faces_count)

                if video_writer:
                    video_writer.write(display_frame)

                cv.imshow('Face Detection - Enhanced', display_frame)

            key: int = cv.waitKey(10 if not paused else 0)

            if key == ord('q') or key == ord('Q'):  # Q to quit
                print("\n⏹ Stopped by user")
                break
            elif key == ord('s') or key == ord('S'):  # Screenshot
                save_screenshot(display_frame, faces_count)
            elif key == ord('p') or key == ord('P'):  # Pause
                paused = not paused
                print("\n⏸ Paused" if paused else "\n▶ Resumed")
            elif key == 32 and paused:  # Space - next frame when paused
                paused = False
                continue

    except KeyboardInterrupt:
        print("\n\n⏹ Interrupted by user")

    end_time: float = time.time()
    duration: float = end_time - start_time

    cap.release()
    if video_writer:
        video_writer.release()
        print(f"\n✓ Video saved to: {args.output}")

    cv.destroyAllWindows()
    print_final_stats(frame_count, duration)