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
            # if file != 'Camera_1.mp4':
            #     continue
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
        color = np.array([30 / 255, 144 / 255, 255 / 255])
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
            np_masks.append(mask)
    return np_masks


def split_video_into_images(video_path, images_path):
    if not os.path.exists(images_path):
        os.mkdir(images_path)
    os.system(f'rm {images_path}/*')
    os.system(f"ffmpeg -i '{video_path}' -q:v 2 -v 2 -start_number 0 {images_path}/'%05d.jpg'")


def main():
    videos_path = 'cvdl_dataset'
    images_path = 'out'
    sam2_checkpoint = "models/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    device = 'cuda'

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    point_picker = PointPicker(images_path)
    videos = get_relevant_videos(videos_path)

    for video_path in videos:
        print(f'Processing {video_path}')
        split_video_into_images(video_path, images_path)
        points = point_picker.pick_points()
        masks = get_masks_from_video(images_path, predictor, points)
        visualize_masks(masks)

if __name__ == '__main__':
    main()