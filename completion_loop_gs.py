from run_generator_gs import generate
import os
import argparse
import glob
import multiprocessing


# Define a custom sorting key function
# def get_sort_key(file_path):
#     filename = os.path.basename(file_path)
#     print(filename)
#     xxxx_part = int(filename.split('_')[0].split('snapshot')[1].split('-')[1][0:-4])  # Extract the snapshot number
#     yyyy_part = filename.split('_')[1].split('.')[0]  # extract seed number
#     return int(xxxx_part), int(yyyy_part)  # Convert to integers for sorting

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

def initiate_completions(checkpoint_dir,image_dir,mask,output_dir,truncation = 1):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort the file list using the custom sorting key
    file_pattern = 'network-snapshot-*.pkl'
    file_list = sorted(glob.glob(os.path.join(checkpoint_dir, file_pattern)))
    # print(file_list)
    file_list = sorted(file_list, key=get_sort_key)
    print(file_list)
    # args = [(file_list[i],image_dir,mask,output_dir,truncation) for i in range(len(file_list))]
    # print(args)
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # pool.map(generate,args)
    for i in range(len(file_list)):
        generate((file_list[i],image_dir,mask,output_dir,truncation))


