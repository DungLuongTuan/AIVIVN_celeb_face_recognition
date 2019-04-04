"""
    reuse code from: https://github.com/tomahim/py-image-dataset-generator
"""

from scipy import ndarray
from skimage import transform
from skimage import util
import skimage as sk
import argparse
import random
import os

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

# dictionary of the transformations functions we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}

# augment data
labels = os.listdir("data/image/train")

for label in labels:
    label_path = os.path.join("data/image/train", label)
    images = os.listdir(label_path)
    if len(images) < 3:
        for image in images:
            image_path = os.path.join(label_path, image)
            image_to_transform = sk.io.imread(image_path)
            if len(images) == 1:
                apply_transformations = available_transformations.keys()
            else:
                apply_transformations = random.sample(list(available_transformations.keys()), random.randint(1, 2))
            
            for key in apply_transformations:
                transformed_image = available_transformations[key](image_to_transform)
                new_file_path = label_path + "/" + image[:-4] + "_" + key + ".png"
                sk.io.imsave(new_file_path, transformed_image)
