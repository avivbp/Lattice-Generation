# Lattice-Generation - Guy Shtainer Aviv Ben Porat Adir Barda Amir Gruber

This is the github repository for Lattice Generation Workshop in Computational Learning.
In this repository you will find all pieces of code used to create our results.

The following is an explanation for how to reproduce the results we show in our report:

1) Training Co-Mod-GAN
   
2) Image Completion Using Co-Mod-GAN:
   In order to complete images using our code, you need to run completion_loop.py after changing the images_path, output_dir_path and 
   checkpoint_path according to your hierarchy.
   If you want to get a KL divergence score for your image completions, you should do the following:
   a. run latticeEnergies.py. This will create a lattice_vectors.m file.
   b. run exactCorrFromLattices.m using the lattice_vectors.m file. you should change the input path according to your hierarchy and change the 
      output path however you like. This will create a file called corrMatrices.m.
   c. run Images_to_KL. This will create a graph of the KL divergence of your image completions as a function of snapshot kimg.
   
3) Image Completion Using FCNN:
4) 
