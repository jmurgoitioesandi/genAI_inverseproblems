import numpy as np
import h5py
import os
import math
import random
import matplotlib.pyplot as plt

def fenics_2Darray_to_nparray(input_array, x_coord, y_coord, x_W, x_H, x_C, width, height):
    """
    This function takes an 1D np array created by Fenics and turns into a 4D numpy array.
    The first dimension refers to the figure number, the second and third are the spatial
    dimensions (x_H and x_W, in this order) and the fourth one is the number of channels (x_C).
    For scalar functions the number of channels is .  
    """

    nparray = np.empty((1, x_H, x_W, x_C), dtype = float)
    input_list = list(input_array)

    for idx_1 in range(x_W * x_H):
        for idx_2 in range(x_C):
            nparray[0, x_H-1-round(y_coord[idx_1 * x_C] / (height/(x_H-1))), \
                round(x_coord[idx_1 * x_C] / (width/(x_W-1))), idx_2] = input_list[idx_1*x_C+idx_2]
    
    return nparray

def store_images_hdf5(images,dirname,img_type):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32) to be stored (THIS IS ACTUALLY A LIST OF IMAGES)
        img_dir      image directory
        file_base    file name base (usually 'training' or 'validation')
    """

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    N = len(images)

    filename = f'{dirname}/{img_type}_data.h5'

    # Create a new HDF5 file
    file = h5py.File(filename, "w")


    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.IEEE_F32BE, data=images
    )

    file.close()


    print('Data saved in ' + filename)
    print(f'... total samples      = {N}')


def noise_addition_image(image, noise_level,noise_type='gaussian'):
    """
    This function adds noise to an image.
    It takes the image, the noise level (noise magnitude) and the type of noise,
    the defaul being gaussian. 
    """
    noise_shape = image.shape
    
    if noise_type == 'gaussian':
        noise = np.random.normal(loc=0.0, scale=noise_level, size=noise_shape)
    else:
        raise Exception('No available noise type mentioned.')

    noisy_image = image + noise
    return noisy_image