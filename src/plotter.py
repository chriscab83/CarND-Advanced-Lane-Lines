import matplotlib.pyplot as plt


def plot_images(imgs, titles, file_path=None, show=True):
    n = len(imgs)
    f, axs = plt.subplots(1, n, figsize=(20, 10))
    for i in range(n):
        axs[i].imshow(imgs[i], cmap='gray')
        axs[i].set_title(titles[i], fontsize=30)

    if file_path is not None:
        f.savefig('./writeup_imgs/' + file_path)

    if show is True:
        plt.show()
