from PIL import Image
import numpy as np
import scipy.io as sio
import glob
import os

# Set the working directory to a specific path
# directory = "/Users/guy/Desktop/StyleGAN2/Generated/kimg481co-mod-gan"
kimg = 529
# directory = '/Users/guy/Desktop/StyleGan2/completed_images/34_kimg' + str(kimg)
# directory = '/Users/guy/Desktop/StyleGan2/completed_images/Extracted_34171_stylegan2'
directory = './completions2'

os.chdir(directory)

# Set files pattern
# file_pattern = 'example*.png'
# file_pattern = 'seed*.png'
# file_pattern = 'series*.png'
file_pattern = 'network-snapshot-*.png'

# Extract their paths
file_list = glob.glob(os.path.join("./", file_pattern))

# Define a custom sorting key function
def get_sort_key(file_path):
    filename = os.path.basename(file_path)
    xxxx_part = int(filename.split('__')[0].split('snapshot')[1].split('-')[1])  # Extract the snapshot number
    yyyy_part = filename.split('__')[1].split('.')[0]  # extract seed number
    return int(xxxx_part), int(yyyy_part)  # Convert to integers for sorting

# Sort the file list using the custom sorting key
file_list = sorted(file_list, key=get_sort_key)

# print(file_list)
# file_list = ['seed7594.png']
lattice_vectors = []

for i in range(len(file_list)):
    file_path = file_list[i]
    file_name = os.path.basename(file_path)
    # print(file_name)
    number = file_name[len('seed'):-len('.png')]
    image = Image.open(file_name)
    image = image.convert('RGB')
    lattice_vector = []
    for j in range(3, 13):
        r, g, b = image.getpixel((2, j))
        lattice_vector.append((r + g + b) / (765 / 3))
    lattice_vectors.append(lattice_vector)
# print(lattice_vectors)
sio.savemat('lattice_vectors.mat', {'lattice_vectors': lattice_vectors})
