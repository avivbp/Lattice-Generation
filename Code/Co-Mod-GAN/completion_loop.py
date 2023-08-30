from run_generator import generate
import os
import argparse

# Path to the directory containing the images
image_dir = "./seeds"

# Path to the directory where you want to save the output images
output_dir = "completions2"

# Path to the mask image
mask = "./mask.jpg"
truncation = 1

# Path to the directory containing the .pkl checkpoint files
checkpoint_dir = "./pkls"

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
