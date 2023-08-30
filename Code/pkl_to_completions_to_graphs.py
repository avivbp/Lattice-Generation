from completion_loop_gs import initiate_completions
from lattice_energies_gs import score_completions
import multiprocessing
import glob
import os
import argparse
import numpy as np


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', help='Network checkpoints directory', required=True)
    parser.add_argument('-i', '--image', help='Original images for completion directory', required=True)
    parser.add_argument('-m', '--mask', help='Mask path', required=True)
    parser.add_argument('-o', '--output', help='Output directory', required=True)
    parser.add_argument('-d', '--done', help='True or Flase, wether complition of al checkpoints is needed or not', required=True)
    parser.add_argument('-t', '--truncation', help='Truncation psi for the trade-off between quality and diversity. Defaults to 1.', default=1)

    args = parser.parse_args()
    if args.done.lower() == "true":
        initiate_completions(args.checkpoint,args.image,args.mask,args.output,args.truncation)
    
    file_pattern = 'pkl-*'
    file_list = sorted(glob.glob(os.path.join(args.output, file_pattern)))
    arguments = [file_list[i] for i in range(len(file_list))]
    print(arguments)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        DKLs = pool.map(score_completions,arguments)
        print(DKLs)
        np.save("DKLs.npy",DKLs)
    
if __name__ == "__main__":
    main()