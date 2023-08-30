import torch
import numpy as np
import EncoderClass as e
import TwoParticlesQW as t
import math

# Instantiate model
model = e.Encoder()

# Load the model's state dictionary from the saved file
checkpoint_path = 'Encoder.pth'
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Function to check conditions and process npy array
def process_npy_array(npy_filename):
    # Load npy array
    array = np.load(npy_filename)
    
    # Check shape and sum condition
    if array.shape == (10, 10):
        # Convert npy array to PyTorch tensor
        tensor_array = torch.from_numpy(array).float()
        
        # Reshape tensor to match model input size if needed
        # Assuming the model input size is (batch_size, channels, height, width)
        tensor_array = tensor_array.unsqueeze(0).unsqueeze(1)  # Adding batch and channel dimensions
        
        # Run the model
        with torch.no_grad():
            output = model(tensor_array)
        
        return output
    else:
        print("Conditions not met for processing the array.")
        return None

def calc_KL_score_symetric(gamma_exact, gamma_estimated, N_particles=2):  # h represents the energies of the 1D lattice and gamma represents the correlation matrix. all as np.array
    L = gamma_exact.shape[0]
    epsilon = (L ** -2) * (10 ** -10)
    return 0.5 * sum([(P + epsilon) * math.log((P + epsilon) / (Q + epsilon)) + (Q + epsilon) * math.log((Q + epsilon) / (P + epsilon)) for P, Q in zip(gamma_estimated.flatten() / N_particles, gamma_exact.flatten() / N_particles)])

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python script_name.py npy_filename.npy")
    else:
        npy_filename = sys.argv[1]
        input_correlation = np.load(npy_filename)
        output = process_npy_array(npy_filename)
        
        if output is not None:
            vector = output.detach().numpy().reshape(10)
            print("Estimated lattice energies required:\n", vector)
            psi0 = np.zeros(55)
            psi0[35] = 1
            output_correlation = t.initial_condition_to_corr(10,vector,np.ones([10,10]),3,psi0,2,return_correlation=True)
            print("Correlation: \n"+str(output_correlation))
            print("KL Divergence with input matrix = "+str(calc_KL_score_symetric(input_correlation,output_correlation)))
        else:
            print("Problem with energy calculation")
