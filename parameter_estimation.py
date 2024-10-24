import os
import cv2 as cv
from sympy.physics.units import current
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def read_video(video_path):
    frames = []
    capture = cv.VideoCapture(video_path)
    fps = capture.get(cv.CAP_PROP_FPS)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frames.append(frame)
    capture.release()
    return frames, fps


def elasticity_estimation(ts_up, ts_down):
    g = 9.81
    assert len(ts_up) > 2, 'Need at least 3 points to estimate elasticity'
    t_down_0 = ts_down[1] - ts_up[1]
    v_down_0 = g * t_down_0

    elasticities = []
    for i in range(2, len(ts_down)):
        t_down_i = ts_down[i] - ts_up[i]
        v_down_i = g * t_down_i
        elasticity_i = (v_down_i / v_down_0) ** (1 / (i - 1))
        elasticities.append(elasticity_i)
    elasticities = [e for e in elasticities if e < 1.0]
    return elasticities

def from_saved_data():
    table_01 = [0.966091783079296, 0.9772648059188248, 0.9036020036098447, 0.9028804514474342, 0.9183859021684457, 0.9296239874987812, 0.9381427059852852, 0.9188042082624673, 0.9266178842891434, 0.9330602913422874]
    table_02 = [0.9309493362512625, 0.9772648059188255, 0.945741609003176, 0.9563524997900368, 0.934655265184067, 0.9296239874987812, 0.9244317242460192, 0.9325379729766575, 0.9390741217932272, 0.9444559403623386, 0.9489643423938208, 0.9319427413981982, 0.9366464979509298, 0.9407422865561211]
    table_03 = [0.966091783079296, 0.9283177667225561, 0.945741609003176, 0.9221079114817281, 0.9183859021684457, 0.9437220574354979, 0.9505798249541406, 0.9448223083691855, 0.9266178842891434, 0.944455940362338, 0.9583245286271477, 0.9319427413981982]
    table_04 = [0.9375000000000002, 0.9354143466934856, 0.9331277892043123, 0.9306048591020999, 0.91028210151304, 0.9246555971486617]
    table_05 = [0.8750000000000003, 0.935414346693485, 0.9331277892043123, 0.9306048591020999, 0.9440875112949022, 0.9246555971486617, 0.9350611267692983, 0.9170040432046712, 0.9380712724561938, 0.9330329915368077, 0.9276019257353447, 0.9334294626458085, 0.9383890415856322, 0.9323385936570352]

    mousepad_01 = [0.9, 0.7745966692414828, 0.6694329500821697]
    mousepad_02 = [0.7500000000000002, 0.7637626158259728, 0.7937005259841]
    mousepad_03 = [0.8, 0.7745966692414837, 0.8434326653017495]
    mousepad_04 = [0.7272727272727271, 0.738548945875996, 0.566516334942704]
    mousepad_05 = [0.5714285714285714, 0.7071067811865477, 0.753947441129154]

    tennis_01 = [0.7857142857142859, 0.7559289460184544, 0.753947441129154, 0.8091067115702214]
    tennis_02 = [0.6428571428571429, 0.8017837257372729, 0.8630543739971823, 0.7730551756939458]
    tennis_03 = [0.642857142857143, 0.7559289460184551, 0.753947441129154]
    tennis_04 = [0.769230769230769, 0.7844645405527361, 0.7272363035371389, 0.7875110621102678]
    tennis_05 = [0.7142857142857137, 0.707106781186547, 0.753947441129154, 0.7730551756939458]

    total = table_01 + table_02 + table_03 + table_04 + table_05
    mean = np.mean(total)
    std = np.std(total)
    minv = np.min(total)
    maxv = np.max(total)
    return mean, std, minv, maxv


def process_bouncing_ball_video(frames, fps):
    current_idx = 0
    ts_up = []
    ts_down = []
    while True:
        frame = frames[current_idx]
        cv.imshow('frame', frame)
        key = cv.waitKey(0) & 0xFF
        if key == ord('p'):
            print(f'Current frame: {current_idx}')
        elif key == ord('d'):
            print(f'Ball touched the ground at frame {current_idx}')
            ts_down.append(current_idx / fps)
        elif key == ord('u'):
            print(f'Reached top at frame {current_idx}')
            ts_up.append(current_idx / fps)
        elif key == ord('e'):
            print(f'Estimating the parameters')
            print(elasticity_estimation(ts_up, ts_down))
        elif key == 83:  # right arrow key
            current_idx = min(current_idx + 1, len(frames) - 1)
            if current_idx == len(frames) - 1:
                print('Reached end of video')
        elif key == 81:  # left arrow key
            current_idx = max(current_idx - 1, 0)
        elif key == ord('q'):
            break
    elasticities = elasticity_estimation(ts_up, ts_down)
    cv.destroyAllWindows()
    return elasticities


def process_led_video(frames, fps):
    # cv.imshow('frame', frames[0])
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    pixel_x = 1026
    pixel_y = 441
    # for frame in frames:
    #     cv.imshow('frame', frame)
    #     frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #     print(np.mean(frame[:, :, 2]))
    #     cv.waitKey(0)
    # frames = [cv.cvtColor(frame, cv.COLOR_BGR2HSV) for frame in frames]
    values = [np.mean(frame[:, :, 2]) for frame in frames]
    plt.plot(values[260:380])
    plt.savefig('output_selected/led/led_experiment_2s_first.png')
    # print(values)


def process_pendulum_video(frames, fps):
    coords_x = []
    for frame in frames:
        coordinates = np.where(frame[:, :, 2] > 127.)
        mean_x = np.mean(coordinates[1])
        mean_y = np.mean(coordinates[0])
        coords_x.append(mean_x)
    coords_x = np.array(coords_x)
    coords_x = coords_x - np.median(coords_x)
    plt.plot(coords_x)
    plt.savefig('tmp/pendulum.png')
    plt.clf()

    peaks = []
    for i in range(5, len(coords_x) - 5):
        if coords_x[i] > coords_x[i - 5] and coords_x[i] > coords_x[i + 5]:
            peaks.append((i / fps, coords_x[i]))
    plt.plot([p[0] for p in peaks], [p[1] for p in peaks], 'ro')
    plt.savefig('tmp/pendulum_peaks.png')
    plt.clf()

    ln_peaks = [(inx, np.log(val)) for (inx, val) in peaks]
    plt.plot([p[0] for p in ln_peaks], [p[1] for p in ln_peaks], 'ro')
    plt.savefig('tmp/pendulum_peaks_log.png')
    plt.clf()

    X = np.array([p[0] for p in ln_peaks]).reshape(-1, 1)
    y = np.array([p[1] for p in ln_peaks])
    reg = LinearRegression().fit(X, y)
    return reg.coef_ * 2  # multiply by 2: https://web.physics.ucsb.edu/~lecturedemonstrations/Composer/Pages/40.37.html


def process_bouncing_ball_videos(videos_path):
    all_elasticities = []
    for index in sorted(os.listdir(videos_path)):
        video_path = os.path.join(videos_path, index, 'video.mp4')
        print(f'Processing video: {video_path}')
        frames, fps = read_video(video_path)
        elasticities = process_bouncing_ball_video(frames, fps)
        all_elasticities.extend(elasticities)
    print(all_elasticities)
    all_elasticities = np.array(all_elasticities)
    mean = np.mean(all_elasticities)
    std = np.std(all_elasticities)
    minv = np.min(all_elasticities)
    maxv = np.max(all_elasticities)
    # mean, std, minv, maxv = from_saved_data()
    print(f'Mean elasticity: {mean}')
    print(f'Std elasticity: {std}')
    print(f'Min elasticity: {minv}')
    print(f'Max elasticity: {maxv}')


def friction_coefficient_estimation(frames, fps, angle):
    angle_rad = np.deg2rad(angle)
    g = 9.81
    s = (81.3 - 8.7) / 100.0  # 81.3 cm - 8.7 cm
    t = len(frames) / fps
    mu = np.tan(angle_rad) - 2 * s / (g * t ** 2)
    return mu


def process_sliding_block_video(frames, fps, angle):
    return friction_coefficient_estimation(frames, fps, angle)


def process_led_videos(videos_path):
    frames, fps = read_video(videos_path)
    process_led_video(frames, fps)

def process_pendulum_videos(videos_path):
    coeffs = []
    for folder in os.listdir(videos_path):
        video_path = os.path.join(videos_path, folder, 'masks.mp4')
        print(f'Processing video: {video_path}')
        frames, fps = read_video(video_path)
        coeff = process_pendulum_video(frames, fps)
        coeffs.append(coeff)
    coeffs = np.array(coeffs)
    mean = np.mean(coeffs)
    std = np.std(coeffs)
    minv = np.min(coeffs)
    maxv = np.max(coeffs)
    print(f'Mean coefficient: {mean}')
    print(f'Std coefficient: {std}')
    print(f'Min coefficient: {minv}')
    print(f'Max coefficient: {maxv}')


def process_sliding_block_videos(videos_path, parameters_save_path):
    with open(parameters_save_path, 'r') as f:
        parameters = json.load(f)
    all_coefficients = []
    for setting in sorted(os.listdir(videos_path)):
        angle = parameters['sliding_block'][setting]['angle']['mean']
        for index in sorted(os.listdir(os.path.join(videos_path, setting))):
            video_path = os.path.join(videos_path, setting, index, 'video.mp4')
            print(f'Processing video: {video_path}')
            frames, fps = read_video(video_path)
            coefficient = process_sliding_block_video(frames, fps, angle)
            all_coefficients.append(coefficient)
    print(all_coefficients)
    mean = np.mean(all_coefficients)
    std = np.std(all_coefficients)
    minv = np.min(all_coefficients)
    maxv = np.max(all_coefficients)
    # mean, std, minv, maxv = from_saved_data()
    print(f'Mean elasticity: {mean}')
    print(f'Std elasticity: {std}')
    print(f'Min elasticity: {minv}')
    print(f'Max elasticity: {maxv}')

if __name__ == '__main__':
    videos_path = 'output_selected/pendulum/pendulum_150'
    parameters_save_path = 'output_selected/parameters.json'
    # process_sliding_block_videos(videos_path, parameters_save_path)
    process_pendulum_videos(videos_path)