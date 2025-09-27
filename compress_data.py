import os
import warnings

import mediapipe as mp
import numpy as np
import skimage.io as io
from tqdm import tqdm

import utils

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="SymbolDatabase.GetPrototype\\(\\) is deprecated",
)


def collect_images(base_path, set_name, version, model=None):
    size = utils.db_size(set_name)

    array = np.zeros((size, 224, 224, 3), dtype=np.uint8)
    lm_shape = (21, 2)
    landmarks_array = np.zeros((size, *lm_shape), dtype=np.float32)

    version_mult = {
        utils.sample_version.gs: 0,
        utils.sample_version.hom: 1,
        utils.sample_version.sample: 2,
        utils.sample_version.auto: 3,
    }[version]
    start = size * version_mult
    end = start + size
    for idx in tqdm(range(start, end), f"Collecting {set_name} {version} images..."):
        img_rgb_path = os.path.join(base_path, set_name, "rgb", "%08d.jpg" % idx)
        img = io.imread(img_rgb_path)
        array[idx] = img

        if model:
            result = model(img)
            if result.multi_hand_landmarks is not None:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks_array[idx] = np.array(
                        [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                    )
            else:
                landmarks_array[idx] = np.full(lm_shape, -1.0, dtype=np.float32)

    return array, landmarks_array


def collect_labels(base_path, set_name):
    k_list, xyz_list = utils.load_db_annotation(base_path, set_name)
    assert len(k_list) == len(xyz_list)

    array = np.zeros((len(xyz_list), 21, 2), dtype=np.float32)

    for idx in tqdm(range(len(xyz_list)), f"Collecting {set_name} labels..."):
        # project camera coords to image plane
        K, xyz = k_list[idx], xyz_list[idx]
        uv = utils.project_points(xyz, K)
        array[idx] = uv.astype(np.float32)

    return array


if __name__ == "__main__":
    data_path = "data"
    ds = "training"
    run_mediapipe = True

    mp_hands = mp.solutions.hands
    hands = (
        mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
        )
        if run_mediapipe
        else None
    )

    gs, lms = collect_images(
        data_path,
        set_name=ds,
        version=utils.sample_version.gs,
        model=lambda x: hands.process(x),
    )
    labels = collect_labels(data_path, set_name=ds)

    np.savez(f"{ds}_dataset.npz", x=gs, y=labels, raw_pred=lms)
