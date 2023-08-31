from scipy.io import loadmat

def extract_numpyArr_from_MATLABArr(matlabFilePath):
    '''
    This function receives the full path of the Matlab file which contains the exact calculation of the correlation
    matrices. In order to extract matrix #d from the returned value "matrices", use the following format: matrices[
    matrix + "%d"][0,0].

        Args:
            matlabFileFolderPath (str): Path to the folder in which corrMatrices.mat file is in.

        Returns:
            matrices: Array of matrices that correspond to the energy lattices.
        '''
    matrices = loadmat(matlabFilePath + '/corrMatrices.mat')
    return matrices['matrix_struct']
