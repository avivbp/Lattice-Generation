from run_generator import generate
import os
import argparse



def initiate_completions(checkpoint_dir,image_dir,mask,output_dir,truncation = 1):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through checkpoint files starting with "network-snapshot"
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("network-snapshot") and filename.endswith(".pkl"):
            print("handling " + filename)
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            
            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Call the generate function
            generate(checkpoint_path, image_dir, mask, output_dir, truncation)
            # os.chdir("..")
            os.rename("./pkls/" + filename,"./handled_pkls/" + filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', help='Network checkpoints directory', required=True)
    parser.add_argument('-i', '--image', help='Original images for completion directory', required=True)
    parser.add_argument('-m', '--mask', help='Mask path', required=True)
    parser.add_argument('-o', '--output', help='Output directory', required=True)
    parser.add_argument('-t', '--truncation', help='Truncation psi for the trade-off between quality and diversity. Defaults to 1.', default=1)

    args = parser.parse_args()
    initiate_completions(**vars(args()


if __name__ == "__main__":
    main()
