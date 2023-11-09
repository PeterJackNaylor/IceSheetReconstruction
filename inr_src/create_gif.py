import glob
from PIL import Image
import os


def stich(date, list_images, path):
    images = [Image.open(os.path.join(path, x).format(date)) for x in list_images]
    widths, heights = zip(*(i.size for i in images))

    img_width = sum(widths) // 5
    img_height = sum(heights) // 5

    new_im = Image.new("RGB", (img_width * 2, img_height * 3))

    positions = [
        (0, 0),
        (img_width, 0),
        (0, img_height),
        (img_width, img_height),
        (img_width // 2, img_height * 2),
    ]
    for i, im in enumerate(images):
        new_im.paste(im, positions[i])
    return new_im


def make_gif(frame_folder, keys, outname):
    files = glob.glob(f"{frame_folder}/{keys[0]}".format("*"))
    files.sort()
    dates = [f.split("siren_")[1].split(".p")[0] for f in files]

    frames = [stich(key, keys, frame_folder) for key in dates]
    frame_one = frames[0]
    frame_one.save(
        outname,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=1000,
        loop=0,
    )


if __name__ == "__main__":
    make_gif(
        "gif/",
        [
            "siren/siren_{}.png",
            "gt_siren/gt_siren_{}.png",
            "siren_mse/mse_siren_{}.png",
            "siren_mae/mae_siren_{}.png",
        ],
        "siren_timeseries.gif",
    )
