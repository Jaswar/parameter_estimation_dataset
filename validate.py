import cv2 as cv
import os

def get_num_frames(video_path):
    capture = cv.VideoCapture(video_path)
    num_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    capture.release()
    return num_frames


def validate(directory):
    for root, dirs, files in os.walk(directory):
        if len(files) == 2:
            assert all(file.endswith('.mp4') for file in files), 'Video should be mp4'

            video_path = os.path.join(root, 'video.mp4')
            masks_path = os.path.join(root, 'masks.mp4')
            num_frames_video = get_num_frames(video_path)
            num_frames_masks = get_num_frames(masks_path)
            assert num_frames_video == num_frames_masks, 'Number of frames should be the same'


def main():
    directory = 'output'
    validate(directory)


if __name__ == '__main__':
    main()

