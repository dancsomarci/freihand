import matplotlib.pyplot as plt

import utils


def show_sample(base_path, idx=1):
    k_list, xyz_list = utils.load_db_annotation(base_path, "training")

    img = utils.read_img(idx, base_path, "training", utils.sample_version.gs)

    # project camera coords to image plane
    K, xyz = k_list[idx], xyz_list[idx]
    uv = utils.project_points(xyz, K)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(img)
    utils.plot_hand(ax1, uv, order="uv")
    plt.show()


if __name__ == "__main__":
    show_sample(
        "data",
        idx=0,
    )
