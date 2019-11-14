import skimage.io as io
import numpy as np

file_location = "/media/tbell/datasets/natural_images.txt"
min_img_pixels = 100
min_img_variance = 1e-4

filenames = [string.strip() for string in open(file_location, "r").readlines()]
num_files = len(filenames)

working_files = []
broken_files = []

for file_id, filename in enumerate(filenames):
  print(file_id, " out of ", num_files, " is ", filename)
  try:
    image = io.imread(filename, as_gray=True)
    if image.size < min_img_pixels or np.var(image) < min_img_variance:
      broken_files.append(filename)
    else:
      working_files.append(filename)
  except:
    broken_files.append(filename)

output_file = "/media/tbell/datasets/verified_images.txt"
with open(output_file, "w") as fi:
  for filename in working_files:
    fi.write("%s\n" % filename)

output_file = "/media/tbell/datasets/bad_images.txt"
with open(output_file, "w") as fi:
  for filename in broken_files:
    fi.write("%s\n" % filename)

import IPython; IPython.embed(); raise SystemExit
