#!/usr/bin/env python3.7

import math, numpy as np
import scipy.io as sio

from PIL import Image
import matplotlib.pyplot as plt

import glob
import os
from scipy.io import loadmat

kimg = 529
# directory = '/Users/guy/Desktop/StyleGan2/completed_images/34_kimg' + str(kimg)
# directory = '/Users/guy/Downloads/1k kimg=23k'
# directory = '/Users/guy/Desktop/StyleGan2/completed_images/Extracted_34171_stylegan2'
directory = 'C:\\Users\\Aviv\\PycharmProjects\\CLWS2\\seeds_first'


def create_image_from_matrix():
    os.chdir(directory)

    # Set files pattern
    # file_pattern = 'series*.png'
    file_pattern = 'sub*.png'
    matrices = loadmat('pngs/corrMatrices.mat')

    # Extract their paths
    file_list = sorted(glob.glob(os.path.join("./", file_pattern)))
    print(file_list)

    for i in range(len(file_list)):
        print(i)
        matrix_num = "matrix" + str(i + 1)
        u_Ref = matrices['matrix_struct'][matrix_num][0, 0]
        u_Ref_for_image = np.power(u_Ref / 2, 0.5)
        print(u_Ref_for_image)

        # Cf = np.sum(image, axis=2) / 765.
        # C = u_Ref_for_image[3:13, 3:13]
        # scaleu = 2 * C ** 2

        cmap = 'YlGnBu_r'

        plt.imshow(u_Ref, cmap=cmap)

        # Add colorbar for reference
        # plt.colorbar()

        # Show the image
        plt.show()


def calc_KL_score_symetric(gamma_exact, gamma_estimated,
                           N_particles=2):  # h represents the energies of the 1D lattice and gamma represents the correlation matrix. all as np.array
    L = gamma_exact.shape[0]
    epsilon = (L ** -2) * (10 ** -10)
    return 0.5 * sum([(P + epsilon) * math.log((P + epsilon) / (Q + epsilon)) + (Q + epsilon) * math.log(
        (Q + epsilon) / (P + epsilon)) for P, Q in
                      zip(gamma_estimated.flatten() / N_particles, gamma_exact.flatten() / N_particles)])


def sort_key(filename):
    name = os.path.basename(filename)
    number = name.split('_')[1].split('.')[0]
    # print(number)
    return int(number)


def RGB_To_h_and_u():
    # Set files pattern
    file_pattern = 'seed_*.png'
    matrices = loadmat('corrMatrices_first.mat')
    # print(matrices["G2"])

    # Extract their paths
    os.chdir(directory)
    seed_list = glob.glob(os.path.join(file_pattern))
    seed_list = sorted(seed_list, key=sort_key)
    print(seed_list)

    kimgs = [24 * i for i in range(43)]
    KL_list = []
    KL_means = []
    KL_stds = []

    for j in range(43):
        print(j)
        for i in range(len(seed_list)):
            file_name = os.path.basename(seed_list[i])
            # print(file_name)
            image = Image.open(file_name)
            image = image.convert('RGB')
            matrix_num = "matrix" + str(i + 1 + j * len(seed_list))
            u_Ref = matrices['matrix_struct'][matrix_num][0, 0]
            Cf = np.sum(image, axis=2) / 765.
            C = Cf[3:13, 3:13]
            scaleu = 2 * C ** 2

            KL = calc_KL_score_symetric(gamma_exact=u_Ref, gamma_estimated=scaleu, N_particles=2)
            KL_list.append(KL)
        KL_mean = np.mean(KL_list)
        KL_std = np.std(KL_list)
        # print(KL_mean)
        # print(KL_std)
        KL_means.append(KL_mean)
        KL_stds.append(KL_std)
        KL_list = []
    # print("kimg=" + str(kimg) + " with new co-mod-gan training ")
    # print("The number of completions made: " + str(len(KL_list)))
    # print("The mean of KLD is:" + str(KL_mean))
    # print("The std of KLD is: " + str(KL_std))
    # print("rel error: " + str(KL_std / KL_mean))
    plt.errorbar(kimgs, KL_means, yerr=KL_stds)
    plt.xlabel('kimg')
    plt.ylabel('KL_divergence')
    plt.title('KL divergence as a function of network kimg')
    plt.show()


def average_of_images():
    directories = ["/Users/guy/Desktop/StyleGAN2/Generated/" + str(i) for i in range(1, 6)]
    for directory in directories:
        os.chdir(directory)
        # file_pattern = 'seed*.png'
        file_pattern = 'series*.png'
        # matrices = loadmat('corrMatrices.mat')

        # Extract their paths
        file_list = sorted(glob.glob(os.path.join("./", file_pattern)))

        file_name = os.path.basename(file_list[0])
        image = Image.open(file_name)
        image = image.convert('RGB')
        matrix_num = "matrix" + str(1)
        # u_Ref = matrices['matrix_struct'][matrix_num][0, 0]

        Cf = np.sum(image, axis=2) / 765.
        C = Cf[3:13, 3:13]
        for i in range(1, len(file_list)):
            file_name = os.path.basename(file_list[i])
            image = Image.open(file_name)
            image = image.convert('RGB')
            # matrix_num = "matrix" + str(i + 1)
            # u_Ref = matrices['matrix_struct'][matrix_num][0, 0]

            Cf = np.sum(image, axis=2) / 765.
            C += Cf[3:13, 3:13]
            # scaleu = 2 * C ** 2

        C = C / len(file_list)

        cmap = 'YlGnBu_r'

        plt.imshow(C, cmap=cmap)

        # Add colorbar for reference
        plt.colorbar()

        # Show the image
        plt.show()


# average_of_images()
RGB_To_h_and_u()
# create_image_from_matrix()
