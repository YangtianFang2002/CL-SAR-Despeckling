from sklearn.model_selection import train_test_split
import glob
import numpy as np
import cv2
import os
import tqdm
import multiprocessing

def add_noise(image, L=None):
    if L is None:
        L = int(np.random.choice([1, 2, 3, 4]))
    assert isinstance(L, int)
    img_max = np.max(image)

    img_size_numpy = image.shape
    rows = img_size_numpy[0]
    columns = img_size_numpy[1]
    s = np.zeros((1, rows, columns))
    for k in range(0, L):
        gamma = np.abs(np.random.randn(1, rows, columns) + np.random.randn(1, rows, columns) * 1j) ** 2 / 2
        s = s + gamma
    s_amplitude = np.sqrt(s / L).squeeze(0)
    if len(image.shape) >= 3:
        s_amplitude = s_amplitude[:, :, np.newaxis]

    noisy_image = np.multiply(image, s_amplitude)
    return np.clip(noisy_image, 0, img_max)

def get_noisy_path(path, key):
    output_parent_path = os.path.dirname(path).replace(key, "Noisy")
    os.makedirs(output_parent_path, exist_ok=True)
    output_path = os.path.join(output_parent_path, os.path.basename(path))
    return output_path

def process_image(args):
    path, replace_key = args
    try:
        sd_ori_gt = cv2.imread(path)
        sd_ori_gt = cv2.cvtColor(sd_ori_gt, cv2.COLOR_BGR2GRAY).astype(np.float32)[:, :, np.newaxis]
        sd_ori_lq = add_noise(sd_ori_gt).astype(np.uint8)  # L=1 is too speckled, HW

        output_path = get_noisy_path(path, replace_key)
        cv2.imwrite(output_path, sd_ori_lq)
        return path, output_path
    except Exception as e:
        return path, None

def main():
    root = "AID"
    replace_key = "Images"
    ext = "jpg"

    paths = glob.glob(root + f"/{replace_key}/**/*.{ext}", recursive=True)

    with multiprocessing.Pool() as pool:
        results = list(tqdm.tqdm(pool.imap_unordered(process_image, [(path, replace_key) for path in paths]), total=len(paths)))

if __name__ == '__main__':
    main()
