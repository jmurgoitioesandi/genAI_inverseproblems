import matplotlib.pyplot as plt


def plotting_image_grid(images, savedir, grid=(4, 4)):
    fig, ax = plt.subplots(grid[0], grid[1], figsize=(grid[0] * 3, grid[1] * 3))

    for i in range(grid[0] * grid[1]):
        ax[i // 4, i % 4].imshow(images[i, 0, :, :])
        ax[i // 4, i % 4].set_xticks([])
        ax[i // 4, i % 4].set_yticks([])

    fig.subplots_adjust(
        wspace=0.04, hspace=0.04, left=0.05, right=0.95, top=0.95, bottom=0.05
    )
    plt.savefig(savedir)
    plt.close()
