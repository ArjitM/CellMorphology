import subprocess
import os
import scipy.io as spio
def makeClusters_Matlab(inFile='/Users/arjitmisra/Documents/Kramer_Lab/Cell-Size-Project/WT/expt_2/piece3-gfp-normal/piece-0010.tif'):
    #eng = matlab.engine.start_matlab()
    #img = eng.imread(inFile.replace('.tif', '_BinaryPivots.tif'))
    subprocess.run("matlab -nosplash -nodesktop -r \"moore_neighbor(\'{0}\'); quit\"".format(inFile.replace('.tif', '_BinaryPivots.tif')), shell=True)
    #os.system("matlab -r \'moore_neighbor(\"{0}\")\'".format(inFile.replace('.tif', '_BinaryPivots.tif')))

    #allBounds = eng.moore_neighbor(img)
    mat = spio.loadmat(inFile.replace('.tif','_bounds.mat'), squeeze_me=True)
    allBounds = mat['all']


makeClusters_Matlab()