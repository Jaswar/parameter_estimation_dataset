import os
import cv2 as cv
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
import math


class PointPicker(object):

    def __init__(self, images_path):
        self.images_path = images_path
        self.points = []
        self.current_frame_idx = 0
        self.current_frame = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.points.append((self.current_frame_idx, [x, y], 1))
            print(f'Positive point {self.current_frame_idx}: {x}, {y}')
        elif event == cv.EVENT_RBUTTONDOWN:
            self.points.append((self.current_frame_idx, [x, y], 0))
            print(f'Negative point {self.current_frame_idx}: {x}, {y}')

    def pick_points(self):
        self.points = []
        self.current_frame_idx = 0
        cv.namedWindow('image')
        files = sorted(os.listdir(self.images_path))
        while True:
            image = cv.imread(os.path.join(self.images_path, files[self.current_frame_idx]))
            cv.imshow('image', image)
            cv.setMouseCallback('image', self.on_mouse)
            key = cv.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == 83:  # right arrow key
                self.current_frame_idx = min(self.current_frame_idx + 1, len(files) - 1)
                if self.current_frame_idx == len(files) - 1:
                    print('Reached end of video')
            elif key == 81:  # left arrow key
                self.current_frame_idx = max(self.current_frame_idx - 1, 0)
        cv.destroyAllWindows()
        return self.points


def get_relevant_videos(videos_path, min_frames=0):
    videos = {}
    for root, dirs, files in os.walk(videos_path):
        for file in files:
            if not file.endswith('.mp4'):
                continue
            path = os.path.join(root, file)
            capture = cv.VideoCapture(path)
            num_frames = capture.get(cv.CAP_PROP_FRAME_COUNT)
            videos[path] = num_frames
            capture.release()
    videos = [video for video, num_frames in videos.items() if num_frames >= min_frames]
    return videos


def visualize_masks(frames, masks):
    for frame, mask in zip(frames, masks):
        color = np.array([0., 1., 0.])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = 0.6 * frame + 0.4 * mask_image
        mask_image = mask_image.astype(np.uint8)
        cv.imshow('mask', mask_image)
        if cv.waitKey(0) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()


def get_masks_from_video_sam2(images_path, predictor, points):
    frame_points = {}
    for point in points:
        if point[0] not in frame_points:
            frame_points[point[0]] = []
        frame_points[point[0]].append(point)
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    np_masks = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=images_path)

        # add new prompts and instantly get the output on the same frame
        for points in frame_points.values():
            frame_idx = points[0][0]
            p = np.array([point[1] for point in points], np.float32)
            labels = np.array([point[2] for point in points], np.int32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=ann_obj_id,
                points=p,
                labels=labels,
            )

        # propagate the prompts to get masklets throughout the video
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask = masks[0].cpu().numpy()
            mask[mask > 0.] = 255.
            mask[mask <= 0.] = 0.
            np_masks.append(mask)
    return np_masks


def get_masks_from_video_bbox(images_path, predictor, points):
    assert len(points) == 2, 'Only two points are supported for bbox'
    first_frame = cv.imread(os.path.join(images_path, '00000.jpg'))
    h, w, _ = first_frame.shape
    minx, miny = points[0][1]
    maxx, maxy = points[1][1]
    num_frames = len(os.listdir(images_path))
    masks = np.zeros((num_frames, 1, h, w), dtype=float)
    masks[:, :, miny:maxy, minx:maxx] = 255.
    return masks


def get_masks_from_video(images_path, predictor, points, use_sam):
    if use_sam:
        return get_masks_from_video_sam2(images_path, predictor, points)
    else:
        return get_masks_from_video_bbox(images_path, predictor, points)


def split_video_into_images(video_path, images_path):
    if not os.path.exists(images_path):
        os.mkdir(images_path)
    os.system(f'rm {images_path}/*')
    os.system(f"ffmpeg -i '{video_path}' -q:v 2 -v 2 -start_number 0 {images_path}/'%05d.jpg'")


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


def save_frames(frames, fps, save_path):
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    video_writer = cv.VideoWriter(save_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def save_frames_and_masks(frames, masks, points, fps, out_path):
    os.makedirs(out_path)

    masks = np.array(masks, dtype=np.uint8)
    masks = np.repeat(masks, 3, axis=1)
    masks = np.transpose(masks, (0, 2, 3, 1))

    frames = np.array(frames, dtype=np.uint8)
    assert frames.shape == masks.shape, \
        f'Shape of frames and masks does not match: {frames.shape} vs {masks.shape}'

    video_save_path = os.path.join(out_path, 'video.mp4')
    save_frames(frames, fps, video_save_path)
    masks_save_path = os.path.join(out_path, 'masks.mp4')
    save_frames(masks, fps, masks_save_path)


def process_chunk(chunk_path, images_path, predictor, point_picker, chunk_out_path, use_sam):
    frames = masks = points = fps = None
    finished = False
    while not finished:
        frames, fps = read_video(chunk_path)
        split_video_into_images(chunk_path, images_path)
        points = point_picker.pick_points()
        masks = get_masks_from_video(images_path, predictor, points, use_sam)

        if use_sam:
            frames = frames[points[0][0]:]  # first point is the starting point
        visualize_masks(frames, masks)

        choice = input('Repeat? (y/n): ')
        if choice == 'n':
            finished = True
    save_frames_and_masks(frames, masks, points, fps, chunk_out_path)


def split_into_chunks(video_path, chunking_path, chunk_length=600):
    if not os.path.exists(chunking_path):
        os.mkdir(chunking_path)
    os.system(f'rm -r {chunking_path}/*')
    capture = cv.VideoCapture(video_path)
    fps = capture.get(cv.CAP_PROP_FPS)

    current_chunk = []
    current_chunk_idx = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        current_chunk.append(frame)
        if len(current_chunk) == chunk_length:
            save_frames(current_chunk, fps, os.path.join(chunking_path, f'video{current_chunk_idx:03d}.mp4'))
            current_chunk = []
            current_chunk_idx += 1
    if len(current_chunk) > 0:
        save_frames(current_chunk, fps, os.path.join(chunking_path, f'video{current_chunk_idx:03d}.mp4'))


def combine_chunks(chunking_path, out_path):
    os.makedirs(out_path)

    chunk_dirs = sorted(os.listdir(chunking_path))
    chunk_dirs = [d for d in chunk_dirs if os.path.isdir(os.path.join(chunking_path, d))]
    for cdir in chunk_dirs:
        os.system(f"echo \"file {os.path.join(cdir, 'video.mp4')}\" >> {chunking_path}/video_input.txt")
        os.system(f"echo \"file {os.path.join(cdir, 'masks.mp4')}\" >> {chunking_path}/masks_input.txt")

    os.system(f'ffmpeg -f concat -i {chunking_path}/video_input.txt -v 2 -c copy {os.path.join(out_path, "video.mp4")}')
    os.system(f'ffmpeg -f concat -i {chunking_path}/masks_input.txt -v 2 -c copy {os.path.join(out_path, "masks.mp4")}')


def process_video(video_path, chunking_path, images_path, predictor, point_picker, out_path, use_sam):
    split_into_chunks(video_path, chunking_path)
    for chunk in sorted(os.listdir(chunking_path)):
        chunk_path = os.path.join(chunking_path, chunk)
        chunk_out_path = chunk_path.split('.')[0]
        process_chunk(chunk_path, images_path, predictor, point_picker, chunk_out_path, use_sam)
    combine_chunks(chunking_path, out_path)


def main():
    videos_path = 'split_clips'
    output_path = 'output'
    images_path = 'images'
    chunking_path = 'tmp'
    sam2_checkpoint = "models/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    use_sam = True

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    point_picker = PointPicker(images_path)
    videos = get_relevant_videos(videos_path)

    for i, video_path in enumerate(videos):
        print(f'Processing {video_path} ({i + 1}/{len(videos)})')

        video_path_structure = os.path.dirname(video_path).split('/')[1:]
        out_path = os.path.join(output_path, *video_path_structure)
        if os.path.exists(out_path):
            print(f'{video_path} already processed')
            continue

        process_video(video_path, chunking_path, images_path, predictor, point_picker, out_path, use_sam)


if __name__ == '__main__':
    main()