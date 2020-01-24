import numpy as np
import PIL.Image
import os
import scipy


def center_crop(img, new_width=None, new_height=None):
    img = np.float32(img)
    width = img.shape[1]
    height = img.shape[0]
    if new_width is None:
        new_width = min(width, height)
    if new_height is None:
        new_height = min(width, height)
    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))
    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))
    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]
    return center_cropped_img

"""Helper-functions to load MSCOCO DB"""
# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_img(src, img_size=False):
   img = scipy.misc.imread(src, mode='RGB')
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size != False:
       img = scipy.misc.imresize(img, img_size)
   return img

def get_files(img_dir):
    files = list_files(img_dir)
    return list(map(lambda x: os.path.join(img_dir,x), files))

def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break
    return files

"""Helper-functions for image manipulation"""
# borrowed from https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/15_Style_Transfer.ipynb

# This function loads an image and returns it as a numpy array of floating-points.
# The image can be automatically resized so the largest of the height or width equals max_size.
# or resized to the given shape
def load_image(filename, shape=None, max_size=None, scale=1):
    image = PIL.Image.open(filename).convert('RGB')

    if max_size is not None:
        factor = float(max_size) / np.max(image.size)
        size = np.array(image.size) * factor

        size = size.astype(int)
        image = image.resize(size, PIL.Image.LANCZOS)  # PIL.Image.LANCZOS is one of resampling filter

    if shape is not None:
        new_shape = [scale*img_dim for img_dim in shape]
        image = image.resize(new_shape, PIL.Image.LANCZOS)  # PIL.Image.LANCZOS is one of resampling filter
        image = center_crop(image, shape[0], shape[1])
        
    return np.float32(image)

# Save an image as a jpeg-file.
# The image is given as a numpy array with pixel-values between 0 and 255.
def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert to bytes.
    image = image.astype(np.uint8)

    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')