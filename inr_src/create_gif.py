import glob
from PIL import Image
import os

def stich(date, l, path):

    images = [Image.open(os.path.join(path, x).format(date)) for x in l]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im

def make_gif(frame_folder, keys, outname):
    files = glob.glob(f"{frame_folder}/{keys[0]}".format("*"))
    files.sort()
    dates = [f.split("siren_")[1].split(".p")[0] for f in files]

    frames = [stich(key, keys, frame_folder) for key in dates]
    frame_one = frames[0]
    frame_one.save(outname, format="GIF", append_images=frames,
               save_all=True, duration=1000, loop=0)
    

if __name__ == "__main__":
    make_gif("gif/siren/", ["siren_{}.png", "gt_siren_{}.png", "mse_siren_{}.png", "mae_siren_{}.png"], "siren_mse_timeseries.gif")