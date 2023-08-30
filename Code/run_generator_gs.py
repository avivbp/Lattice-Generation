import argparse
import numpy as np
import PIL.Image
import os
import glob

from dnnlib import tflib
from training import misc

def generate(args):
    checkpoint,images_dir,mask,output_dir,truncation = args
    tflib.init_tf()
    
    # os.chdir("./")
    # print("test")
    # print(os.getcwd())
    mask = np.asarray(PIL.Image.open(mask).convert('1'), dtype=np.float32)[np.newaxis]
    
    # Create the output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)
    
    # pkl_list = os.listdir(checkpoint_dir)
    
    filename = os.path.basename(checkpoint)
    print(filename)
    
    
    if filename.startswith("network-snapshot") and filename.endswith(".pkl"):
        print("handling " + filename)
        # checkpoint_path = os.path.join(checkpoint_dir, filename)
        _, _, Gs = misc.load_pkl(checkpoint)

        # Create the output directory for pkl if it doesn't exist
        pkl_number = filename[len("network-snapshot"):-len(".pkl")]
        # print(os.getcwd())
        os.chdir(output_dir)
        # print(os.getcwd())
        os.makedirs("pkl" + pkl_number, exist_ok=True)
        os.chdir("../")
        # print(os.getcwd())

        # os.chdir(images_dir)
        # print(os.getcwd())
        file_pattern = 'seed*.png'
        # Extract their paths
        file_list = sorted(glob.glob(os.path.join(images_dir, file_pattern)))

        # Call the generate function
        for image in file_list:
            latent = np.random.randn(1, *Gs.input_shape[1:])
            real = np.asarray(PIL.Image.open(image)).transpose([2, 0, 1])
            real = misc.adjust_dynamic_range(real, [0, 255], [-1, 1])
            fake = Gs.run(latent, None, real[np.newaxis], mask[np.newaxis], truncation_psi=truncation)[0]
            fake = misc.adjust_dynamic_range(fake, [-1, 1], [0, 255])
            fake = fake.clip(0, 255).astype(np.uint8).transpose([1, 2, 0])
            fake = PIL.Image.fromarray(fake)
            file_name = os.path.basename(image)

            base_filename = os.path.basename(checkpoint)
            number = file_name[len("seed"):-len('.png')]
            snapshot_num = os.path.splitext(base_filename)[0]
            output_filename = "./" + output_dir[1:len(output_dir)] + "/" + "pkl" + pkl_number + "/" + "completion" + str(number) + ".png"
            # output_filename = output_dir + "/" + snapshot_num + "_" + str(number) + ".png"
            fake.save(output_filename)
        # os.chdir("../")

        # os.rename("/results/pkls/" + base_filename,"/results/handled_pkls/" + base_filename)
    # print("finished with seed " + str(number))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', help='Network checkpoint path', required=True)
    parser.add_argument('-i', '--image', help='Original image path', required=True)
    parser.add_argument('-m', '--mask', help='Mask path', required=True)
    parser.add_argument('-o', '--output', help='Output (inpainted) image path', required=True)
    parser.add_argument('-t', '--truncation', help='Truncation psi for the trade-off between quality and diversity. Defaults to 1.', default=None)

    args = parser.parse_args()
    generate(**vars(args))

if __name__ == "__main__":
    main()
