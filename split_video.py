import os
import cv2 as cv
from sympy.physics.units import current
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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
            if current_idx == len(frames) - 1:
                print('Reached end of video')
        elif key == 81:  # left arrow key
            current_idx = max(current_idx - 1, 0)
        elif key == ord('q'):
            break
    cv.destroyAllWindows()
    return clips


def split_led_video(frames):
    values = [np.mean(cv.cvtColor(frame, cv.COLOR_BGR2HSV)[:, :, 2]) for frame in frames]

    print(f'Total frames: {len(values)}')
    clips = []
    while True:
        start = int(input('Start: '))
        if start > len(values):
            break
        end = int(input('End: '))
        to_display = values[start:end]
        plt.figure(figsize=(40, 20))
        plt.plot(to_display)
        plt.show()
        print(f'Clip length: {len(to_display)}')
        keep = input('Keep? (y/n): ') == 'y'
        if keep:
            print('Clip created')
            clips.append(frames[start:end])
    return clips



def save_clips(clips, fps, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)
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
    matplotlib.use('tkagg')

    video_path = 'recordings/new/dropped_ball_large.mp4'
    save_directory = 'split_clips/dropped_ball/large'

    frames, fps = read_video(video_path)
    print(fps)
    clips = split_video(frames)
    save_clips(clips, fps, save_directory)