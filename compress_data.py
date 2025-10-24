import os
import pickle
import time

import lmdb
import numpy as np
import skimage.io as io
from tqdm import tqdm

import utils


def loop_images(base_path, set_name, version):
    size = utils.db_size(set_name)

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
        yield img  # final shape: (size, 224, 224, 3) uint8


def loop_labels(base_path, set_name):
    k_list, xyz_list = utils.load_db_annotation(base_path, set_name)
    assert len(k_list) == len(xyz_list)

    for idx in range(len(xyz_list)):
        # project camera coords to image plane
        K, xyz = k_list[idx], xyz_list[idx]
        uv = utils.project_points(xyz, K)
        yield uv.astype(np.float32)  # final shape: (len(xyz_list), 21, 2) float32


if __name__ == "__main__":
    # TODO: set parameters here
    data_path = "data"
    ds = "training"
    map_size_gb = 4

    images = loop_images(
        data_path,
        set_name=ds,
        version=utils.sample_version.gs,
    )
    labels = loop_labels(data_path, set_name=ds)

    output_path = f"{ds}_data_{time.time()}.lmdb"
    map_size = int(map_size_gb * 1024 * 1024 * 1024)  # Convert GB → bytes

    idx = 0
    with lmdb.open(output_path, map_size=map_size) as env:
        with env.begin(write=True) as txn:
            for img, label in zip(images, labels):
                sample = {"image": img, "label": label}
                key = f"{idx:08d}".encode("ascii")
                value = pickle.dumps(sample)
                txn.put(key, value)
                idx += 1

    env.close()
    print(f"✅ LMDB dataset created at {output_path}")
