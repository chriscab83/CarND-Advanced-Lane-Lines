import matplotlib.pyplot as plt


def plot_images(imgs, titles, file_path=None, show=True):
    n = len(imgs)
    if n == 1:
        plot_image(imgs[0], titles[0], file_path, show)
        return

    f, axs = plt.subplots(1, n, figsize=(20, 10))
    for i in range(n):
        axs[i].imshow(imgs[i], cmap='gray')
        axs[i].set_title(titles[i], fontsize=30)

    if file_path is not None:
        f.savefig('./writeup_imgs/' + file_path)

    if show is True:
        plt.show()

    plt.close()


def plot_image(img, title, file_path=None, show=True):
    f, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(img, cmap='gray')
    ax.set_title(title, fontsize=30)

    if file_path is not None:
        f.savefig('./writeup_imgs/' + file_path)

    if show is True:
        plt.show()

    plt.close()
