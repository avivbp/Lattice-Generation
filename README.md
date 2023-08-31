# Lattice-Generation - Guy Shtainer Aviv Ben Porat Adir Barda Amir Gruber

This is the github repository for Lattice Generation Workshop in Computational Learning.
In this repository you will find all pieces of code used to create our results.

The following is an explanation for how to reproduce the results we show in our report:

1) For StyleGAN2 training and generating new images, see https://github.com/NVlabs/stylegan2. We ran the code on the 
   university cluster, using the commands in StyleGAN_Docker.txt. This opens a container with the required dependencies for 
   running the code.
2) Training Co-Mod-GAN:
   In order to train Co-Mod-GAN, we used Google Cloud. In there we created a notebook (type in searchbar vertex ai-> go to workbench on side 
   tab and click on new notebook, setup notebook according to your requirements. In a terminal run the commands shown in Co_Mod_Docker.txt, this will 
   enter a container with all required dependencies for running Co-Mod-GAN code, and run the training code which creates snapshots of your 
   network in the from of.pkl files.
   
3) Image Completion Using Co-Mod-GAN:
   Firstly you need to change the run_generator.py file 
   In order to complete images using our code, you need to run completion_loop.py after changing the images_path, output_dir_path and 
   checkpoint_path according to your hierarchy.
   If you want to get a KL divergence score for your image completions, you should do the following:
   a. run latticeEnergies.py. This will create a lattice_vectors.m file.
   
   b. run exactCorrFromLattices.m using the lattice_vectors.m file. you should change the input path according to your hierarchy and change the 
      output path however you like. This will create a file called corrMatrices.m.
   
   c. In order to use the correlation matrices found in the corrMatrices.mat file, use the extract_numpyArr_from_MATLABArr.py python code and provide it the path of corrMatrices.mat.
   
5) Image Completion Using FCNN:
   To generate an energy vector, run
python FindLattice.py <matrix.npy>
Where matrix.npy is a numpy array containing one 10x10 matrix (check example_correlation.npy for an example).
To denoise an input matrix, run
python DenoiseMatrix.py <matrix.npy>
Where matrix.npy is similar to our description above.
To train a new model based on our architecture, use EncoderClass.py as the model class.
Encoder.pth and Denoiser.pth are our best trained models for finding energy vectors and denoising matrices. Check our report for more elaboration on their training process. They can be used directly to predict the correct vector using PyTorch.

