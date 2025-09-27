import matplotlib.pyplot as plt

import utils


def show_sample(base_path, idx=1):
    k_list, xyz_list = utils.load_db_annotation(base_path, "training")

    img = utils.read_img(idx, base_path, "training", utils.sample_version.gs)
    msk = utils.read_msk(idx, base_path)

    # project camera coords to image plane
    K, xyz = k_list[idx], xyz_list[idx]
    uv = utils.project_points(xyz, K)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(img)
    ax2.imshow(msk)
    utils.plot_hand(ax1, uv, order="uv")
    utils.plot_hand(ax2, uv, order="uv")
    ax1.axis("off")
    ax2.axis("off")
    plt.show()


if __name__ == "__main__":
    show_sample(
        "data",
        idx=0,
    )
