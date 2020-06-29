import glob
from PIL import Image
from utils.gen_utils import get_project_root
import re

img_folder = f'{get_project_root()}/img/ppa/ppa_ramping_drive/'
fp_in = f"{img_folder}/" + r"*.png"
fp_out = f"{img_folder}/ppa_evolution_short.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))[::5]]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=100, loop=0)