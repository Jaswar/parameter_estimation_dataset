import os
import cv2 as cv


def read_video(video_path):
    frames = []
    capture = cv.VideoCapture(video_path)
    fps = capture.get(cv.CAP_PROP_FPS)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


def split_video(frames):
    clips = []
    current_idx = 0
    start_idx = 0
    end_idx = 0
    while True:
        frame = frames[current_idx]
        cv.imshow('frame', frame)
        key = cv.waitKey(0) & 0xFF
        if key == ord('s'):
            print(f'Starting clip at frame {current_idx}')
            start_idx = current_idx
        elif key == ord('e'):
            print(f'Ending clip at frame {current_idx}')
            end_idx = current_idx
        elif key == ord('d'):
            clips.append(frames[start_idx:end_idx + 1])
            print(f'Created clip with {end_idx - start_idx + 1} frames')
        elif key == 83:  # right arrow key
            current_idx = min(current_idx + 1, len(frames) - 1)
        elif key == 81:  # left arrow key
            current_idx = max(current_idx - 1, 0)
        elif key == ord('q'):
            break
    cv.destroyAllWindows()
    return clips


def save_clips(clips, fps, save_directory):
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    for i, clip in enumerate(clips):
        save_dir = os.path.join(save_directory, '{:02d}'.format(i + 1))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'video.mp4')
        fourcc = cv.VideoWriter_fourcc(*'MP4V')
        video_writer = cv.VideoWriter(save_path, fourcc, fps, (clip[0].shape[1], clip[0].shape[0]))
        for frame in clip:
            video_writer.write(frame)
        video_writer.release()


if __name__ == '__main__':
    video_path = 'recordings/bouncing_ball_table.mp4'
    save_directory = 'split_clips'

    frames, fps = read_video(video_path)
    clips = split_video(frames)
    save_clips(clips, fps, save_directory)