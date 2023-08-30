from PIL import Image
import numpy as np
import scipy.io as sio
import glob
import os
from TwoParticlesQW import initial_condition_to_corr
# from images_to_KL import RGB_To_h_and_u
import math
import subprocess
import matlab


def calc_KL_score_symetric(gamma_exact, gamma_estimated,N_particles=2):  # h represents the energies of the 1D lattice and gamma represents the correlation matrix. all as np.array
    L = gamma_exact.shape[0]
    epsilon = (L ** -2) * (10 ** -10)
    return 0.5 * sum([(P + epsilon) * math.log((P + epsilon) / (Q + epsilon)) + (Q + epsilon) * math.log(
    (Q + epsilon) / (P + epsilon)) for P, Q in
                    zip(gamma_estimated.flatten() / N_particles, gamma_exact.flatten() / N_particles)])

# Define a custom sorting key function
def get_sort_key(file_path):
    filename = os.path.basename(file_path)
    # print(filename)

    # Split the filename using underscores
    parts = filename.split('_')

    if len(parts) < 2:
        # Handle the case where there are not enough parts
        return 0, 0

    # Extract the snapshot number
    xxxx_part = parts[0].split('snapshot')[1].split('-')[1][0:-4]

    # Extract the seed number
    yyyy_part = parts[1].split('.')[0]

    try:
        return int(xxxx_part), int(yyyy_part)
    except ValueError:
        # Handle the case where xxxx_part or yyyy_part cannot be converted to integers
        return 0, 0
    
def rewrite_matlab_script(path,directory):
    # Calculate accurate matrices
    # Define paths
    matlab_script_path = "/home/co-mod-gan/exactCorrFromLattices.m"  # Replace with the actual path

    # Read the original MATLAB script
    with open(matlab_script_path, "r") as f:
        script_lines = f.readlines()

    # Modify the desired line (line 62 in this case)
    line_number_to_modify = 1
    custom_line_content = 'path = "/home/co-mod-gan/" ' + directory +  'lattice_vectors.mat";\n'
    script_lines[line_number_to_modify - 1] = custom_line_content
    line_number_to_modify = 61
    custom_line_content = 'path = "/home/co-mod-gan/' + directory + '";\n'
    script_lines[line_number_to_modify - 1] = custom_line_content

    # Write the modified script back to the same file
    new_matlab_script_path = "/exactCorrFromLattices.m"
    # print(os.getcwd())
    with open(new_matlab_script_path, "w") as f:
        f.writelines(script_lines)

    # Run the modified MATLAB script using subprocess
    try:
        subprocess.run(["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r", f"run('{new_matlab_script_path}')"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error running modified MATLAB script:", e)

def score_completions(directory):

    # Set the working directory to a specific path
    os.chdir(directory)

    # Set files pattern
    file_pattern = 'completion*.png'

    # Extract their paths
    file_list = glob.glob(os.path.join("./", file_pattern))

    # Sort the file list using the custom sorting key
    file_list = sorted(glob.glob(os.path.join("./", file_pattern)))
    file_list = sorted(file_list, key=get_sort_key)

    # Set initial function
    psi0 = np.zeros(55)
    psi0[35] = 1
    
    # Define importand arrays
    lattice_vectors = []
    CM = []
    DKLs = []
    KL_list = []

    # Run over completions and determine a DKL score for each, and average them and returns.
    print("number of files: " + str(len(file_list)))
    for i in range(len(file_list)):
        file_path = file_list[i]
        file_name = os.path.basename(file_path)
        number = file_name[len('completion'):-len('.png')]
        image = Image.open(file_name)
        image = image.convert('RGB')
        lattice_vector = []
        for j in range(3, 13):
            r, g, b = image.getpixel((2, j))
            lattice_vector.append((r + g + b) / (765 / 3))
        CM.append(initial_condition_to_corr(10, lattice_vector, np.ones([10,10]), 3, psi0, prop_time = 2, plot_result = False, plot_path="", return_correlation=True))
        lattice_vectors.append(lattice_vector)
    
    # Save vectors as MATLAB file
    # sio.savemat('lattice_vectors.mat', {'lattice_vectors': lattice_vectors})
    # rewrite_matlab_script('lattice_vectors.mat',directory)
        
    Cf = np.sum(image, axis=2) / 765.
    C = Cf[3:13, 3:13]
    scaleu = 2*C ** 2
    print("finished with: " + str(i))

    KL = calc_KL_score_symetric(gamma_exact= CM[-1], gamma_estimated=scaleu, N_particles=2)
    KL_list.append(KL)
    KL_mean = np.mean(KL_list)
    KL_std = np.std(KL_list)
    print("kimg=" + str(kimg) + " with new co-mod-gan training ")
    print("The number of completions made: " + str(len(KL_list)))
    print("The mean of KLD is:" + str(KL_mean))
    print("The std of KLD is: " + str(KL_std))
    print("rel error: " + str(KL_std/KL_mean))
    return (KL_mean,KL_std)
    

# score_completions("./completions3/pkl-002622")


