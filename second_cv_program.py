from __future__ import print_function
import cv2 as cv
import argparse
import time
import os
import sys

# Глобальные переменные для статистики
total_faces: int = 0
total_eyes: int = 0
frame_count: int = 0
start_time = None
fps_list = []

def get_video_info(cap):
    """Получает и выводит информацию о видео"""
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    codec = int(cap.get(cv.CAP_PROP_FOURCC))
    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

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


def detectAndDisplay(frame):
    """Обнаруживает лица и глаза, возвращает статистику"""
    global total_faces, total_eyes, frame_count

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    faces_in_frame = len(faces)
    eyes_in_frame = 0

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)

        cv.putText(frame, f"Face {faces_in_frame}", (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        faceROI = frame_gray[y:y + h, x:x + w]

        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
            eyes_in_frame += 1

    total_faces += faces_in_frame
    total_eyes += eyes_in_frame
    frame_count += 1

    return frame, faces_in_frame, eyes_in_frame


def draw_stats(frame, faces_count, eyes_count, current_fps):
    """Рисует статистику на кадре"""
    height, width = frame.shape[:2]

    overlay = frame.copy()
    cv.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    stats = [
        f"Frame: {frame_count}",
        f"FPS: {current_fps:.1f}",
        f"Faces detected: {faces_count}",
        f"Eyes detected: {eyes_count}",
        f"Total faces: {total_faces}",
    ]

    y_offset = 35
    for i, stat in enumerate(stats):
        cv.putText(frame, stat, (20, y_offset + i * 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def save_screenshot(frame, faces_count):
    """Сохраняет скриншот при обнаружении лиц"""
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"screenshots/face_detection_{timestamp}_faces{faces_count}.jpg"
    cv.imwrite(filename, frame)
    print(f"\n📸 Screenshot saved: {filename}")


def print_final_stats(total_frames, duration):
    """Выводит финальную статистику"""
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
    available_cameras = []
    for i in range(10):
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera {i}: Working (resolution: {frame.shape})")
                available_cameras.append(i)
            else:
                print(f"⚠ Camera {i}: Opened but can't read frames")
            cap.release()

    if available_cameras:
        print(f"\n✓ Found {len(available_cameras)} working camera(s): {available_cameras}")
    else:
        print("\n⚠ No working cameras found. Use video file instead.")

    parser = argparse.ArgumentParser(
        description='Enhanced Face Detection with Video Analytics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      python script.py --source video.mp4
      python script.py --source 0 --save-screenshots
      python script.py --source video.mp4 --skip-frames 2
        """
    )
    parser.add_argument('--face_cascade', help='Path to face cascade.',
                        default='cv_haarcascade_xmls/haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.',
                        default='cv_haarcascade_xmls/haarcascade_eye_tree_eyeglasses.xml')
    parser.add_argument('--source', help='Video file or camera index',
                        default='videos/business195.mov')
    parser.add_argument('--save-screenshots', action='store_true',
                        help='Auto-save screenshots when faces detected')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='Skip N frames between processing (for speed)')
    parser.add_argument('--output', help='Save processed video to file')
    args = parser.parse_args()

    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()

    if not face_cascade.load(cv.samples.findFile(args.face_cascade)):
        print('❌ Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv.samples.findFile(args.eyes_cascade)):
        print('❌ Error loading eyes cascade')
        exit(0)

    print("✓ Cascades loaded successfully\n")

    # Открытие источника видео
    try:
        source = int(args.source)
        print(f"📹 Opening camera {source}...")
    except ValueError:
        source = args.source
        print(f"📁 Opening video file: {source}...")

    cap = cv.VideoCapture(source)

    if not cap.isOpened():
        print(f'❌ Error: Could not open {source}')
        exit(0)

    print(f"✓ Successfully opened: {source}")

    # Получение информации о видео
    width, height, video_fps, total_video_frames = get_video_info(cap)

    # Настройка записи видео
    video_writer = None
    if args.output:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(args.output, fourcc, video_fps, (width, height))
        print(f"💾 Recording output to: {args.output}\n")

    start_time = time.time()
    frame_time = time.time()
    paused = False
    skip_counter = 0

    print("🚀 Starting face detection... Press Q to exit\n")

    try:
        while True:
            if not paused:
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
                processed_frame, faces_count, eyes_count = detectAndDisplay(frame)

                # Расчет FPS
                current_time = time.time()
                current_fps = 1 / (current_time - frame_time) if (current_time - frame_time) > 0 else 0
                frame_time = current_time
                fps_list.append(current_fps)

                display_frame = draw_stats(processed_frame, faces_count, eyes_count, current_fps)

                if args.save_screenshots and faces_count > 0 and frame_count % 30 == 0:
                    save_screenshot(display_frame, faces_count)

                if video_writer:
                    video_writer.write(display_frame)

                cv.imshow('Face Detection - Enhanced', display_frame)

            key = cv.waitKey(10 if not paused else 0)

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

    end_time = time.time()
    duration = end_time - start_time

    cap.release()
    if video_writer:
        video_writer.release()
        print(f"\n✓ Video saved to: {args.output}")

    cv.destroyAllWindows()
    print_final_stats(frame_count, duration)