import os
import cv2 as cv
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor


class PointPicker(object):

    def __init__(self, images_path):
        self.images_path = images_path
        self.points = []
        self.current_frame_idx = 0
        self.current_frame = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.points.append((self.current_frame_idx, [x, y]))
            print(f'Point {self.current_frame_idx}: {x}, {y}')

    def pick_points(self):
        self.points = []
        self.current_frame_idx = 0
        cv.namedWindow('image')
        for file in sorted(os.listdir(self.images_path)):
            image = cv.imread(os.path.join(self.images_path, file))
            cv.imshow('image', image)
            cv.setMouseCallback('image', self.on_mouse)
            if cv.waitKey(0) & 0xFF == ord('q'):
                break
            self.current_frame_idx += 1
        cv.destroyAllWindows()
        return self.points


def get_relevant_videos(videos_path, min_frames=60):
    videos = {}
    for root, dirs, files in os.walk(videos_path):
        for file in files:
            if file != 'Camera_1.mp4':
                continue
            if not file.endswith('.mp4'):
                continue
            path = os.path.join(root, file)
            if 'multi' in path or 'liquid' in path:
                continue
            capture = cv.VideoCapture(path)
            num_frames = capture.get(cv.CAP_PROP_FRAME_COUNT)
            videos[path] = num_frames
            capture.release()
    videos = [video for video, num_frames in videos.items() if num_frames >= min_frames]
    return videos


def visualize_masks(masks):
    for mask in masks:
        color = np.array([1., 1., 1.])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        cv.imshow('mask', mask_image)
        if cv.waitKey(0) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()


def get_masks_from_video(images_path, predictor, points):
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    np_masks = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=images_path)

        # add new prompts and instantly get the output on the same frame
        for point in points:
            p = np.array([point[1]], np.float32)
            labels = np.array([1], np.int32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=point[0],
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


def save_video_and_masks(video_path, masks, points, out_path):
    os.makedirs(out_path)

    masks = np.array(masks, dtype=np.uint8)
    masks = np.repeat(masks, 3, axis=1)
    masks = np.transpose(masks, (0, 2, 3, 1))

    frames, fps = read_video(video_path)
    frames = frames[points[0][0]:]  # the first picked point is the starting point
    frames = np.array(frames, dtype=np.uint8)
    assert frames.shape == masks.shape, \
        f'Shape of masks and frames does not match: {frames.shape} vs {masks.shape}'

    video_save_path = os.path.join(out_path, 'video.mp4')
    save_frames(frames, fps, video_save_path)
    masks_save_path = os.path.join(out_path, 'masks.mp4')
    save_frames(masks, fps, masks_save_path)


def process_video(video_path, images_path, predictor, point_picker, out_path):
    finished = False
    masks = None
    points = None
    while not finished:
        print(f'Processing {video_path}')
        split_video_into_images(video_path, images_path)
        points = point_picker.pick_points()
        masks = get_masks_from_video(images_path, predictor, points)
        visualize_masks(masks)

        choice = input('Repeat? (y/n): ')
        if choice == 'n':
            finished = True
    save_video_and_masks(video_path, masks, points, out_path)


def main():
    videos_path = 'phys101'
    output_path = 'output'
    images_path = 'images'
    sam2_checkpoint = "models/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    point_picker = PointPicker(images_path)
    videos = get_relevant_videos(videos_path)

    for video_path in videos:
        video_path_structure = os.path.dirname(video_path).split('/')[1:]
        out_path = os.path.join(output_path, *video_path_structure)
        if os.path.exists(out_path):
            print(f'{video_path} already processed')
            continue

        process_video(video_path, images_path, predictor, point_picker, out_path)


if __name__ == '__main__':
    main()