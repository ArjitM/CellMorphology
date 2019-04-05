import multiprocessing
from multiprocessing import Pool

import Binarize
import Cell_objects
from Binarize import *
from Cell_objects import *
from Stack_objects import *
import numpy as np
from skimage import util
from skimage.filters import threshold_local, threshold_otsu
import subprocess
import copy
import argparse
import pickle
import logging
import traceback
import os
#import matlab.engine
from operator import add
import scipy.ndimage
import scipy.io as spio
from sklearn.cluster import KMeans
import sys
sys.setrecursionlimit(10000)



def makeClusters_Matlab(binary, inFile, stack_slice):
    Cluster.clusters = []
    # eng = matlab.engine.start_matlab()
    # img = eng.imread(inFile.replace('.tif', '_BinaryPivots.tif'))
    # #boundaries = eng.bwboundaries(img)
    # allBounds = eng.moore_neighbor(img)
    #print(allBounds)
    subprocess.run("matlab -nosplash -nodesktop -r \"moore_neighbor(\'{0}\'); quit\"".format(inFile.replace('.tif', '_BinaryPivots.tif')), shell=True)
    mat = spio.loadmat(inFile.replace('.tif','_bounds.mat'), squeeze_me=True)
    allBounds = mat['all']
    clusterBounds = []
    pivots = []
    for boundaries in allBounds:
        k = 0
        if type(boundaries) != np.object:
            boundaries = [boundaries]
        for bound in boundaries:
            pivots.append([])
            if type(bound[0][0]) != np.ndarray:
                clusterBounds.append([tuple( map( add, bp, (-1, -1) ) ) for bp in bound]) #convert from matlab double
            else:
                clusterBounds.append([tuple( map( add, bp, (-1, -1) ) ) for bp in bound[0]])
                for k in range(1, len(bound)):
                    pivots[-1].append([tuple( map( add, bp, (-1, -1) ) ) for bp in bound[k]]) #np.array(bp).astype(np.uint16)
    if not clusterBounds:
        return None, None
    boundary = [item for sublist in clusterBounds for item in sublist] + [item for sublist in pivots for item in sublist]
    Cluster.clusters = [] #VERY IMPORTANT TO RESET
    k=0
    for c, p in zip(clusterBounds, pivots):
        Cluster(binary, c, stack_slice, p)
        k += 1
    print("NUMBER OF CLUSTERS IS: ", len(Cluster.clusters))
    visualize_Clusters(binary, inFile, Cluster.clusters)
    # visualPivots = showPivots(copy.deepcopy(binary), Cluster.clusters)
    # skimage.external.tifffile.imsave(inFile.replace('.tif', '_BinaryPivotPoints.tif'), visualPivots)
    # skimage.external.tifffile.imsave(inFile.replace('.tif', '_BinaryPivotCheck.tif'), binary)
    return Cluster.clusters, boundary

def makeBinary(inFile, pic_array, pic):
    #pic_array = pic.asarray()
    #out_array = pic.asarray(); #copy dimensions
    global WHITE
    WHITE = skimage.dtype_limits(pic_array, True)[1]
    Binarize.WHITE = WHITE
    Binarize.bin_WHITE = WHITE
    Cell_objects.WHITE = WHITE
    out_array = [ [0] * len(pic_array[1]) ] * len(pic_array)
    regions = Compartmentalize(pic_array, 32)
    #out_array = copy.deepcopy(pic_array) #
    out_array = np.array(out_array, dtype=pic_array.dtype.type) # keep same image type
    #basicEdge(pic_array, out_array, regions) # preliminary edge detection via pixel gradient
    noise_handler = Noise(out_array, regions, iterations=3, binary=False)
    noise_handler.reduce() 
    out_array = pic_array > threshold_local(pic_array, block_size=35).astype(pic_array.dtype.type) 
    out_array = out_array.astype(pic_array.dtype.type)
    out_array = np.array([ [WHITE if p else 0 for p in row] for row in out_array], dtype = pic_array.dtype.type)
    if not nucleusMode and not virus_inject:
        out_array = skimage.util.invert(out_array) #inversion required for soma stain
        # out_array = growCellBoundaries(pic_array, out_array)
        # out_array = pic_array < threshold_local(pic_array, block_size=35).astype(pic_array.dtype.type) 
        # out_array = out_array.astype(pic_array.dtype.type)
        # out_array = np.array([ [WHITE if p else 0 for p in row] for row in out_array], dtype = pic_array.dtype.type)
    skimage.external.tifffile.imsave(inFile.replace('.tif', '_edgeBasic.tif'), out_array)
    #regions.setNoiseCompartments(out_array, 0.95)
    #regions.reviseNoiseCompartments()
    for i in range(len(out_array)):
        for j in range(len(out_array[0])):
            if regions.getCompartment(i, j).noise_compartment:
                out_array[i][j] = 0
    #enhanceEdges(pic_array, out_array, regions, nucleusMode=False) # use detected averages to guess missing edges
    #skimage.external.tifffile.imsave(inFile.replace('.tif', '_edgeEnhance.tif'), out_array)
    noise_handler = Noise(out_array, iterations=3, binary=True)
    noise_handler.reduce(largeNoise=True) #reduce salt and pepper noise incorrectly labeled as edges 
    noise_handler.reduce(largeNoise=False)
    skimage.external.tifffile.imsave(inFile.replace('.tif', '_Binary.tif'), out_array)
    print("***made binary")
    labeled, num_objects = scipy.ndimage.label(out_array)
    #print("Objects detected", num_objects)
    #print(labeled)
    #labeled = remove_largeNoise(labeled, num_objects)
    visualize_labeled(labeled, inFile)
    outFile = open(inFile.replace('.tif', '_labeled.pkl'), 'wb')
    pickle.dump(labeled, outFile, protocol=2)
    outFile.close()
    # Cellprofiler and dependencies are built in python2, therefore use pickle for variable handoffs
    os.system('python2 declump_bridge.py "{0}" "{1}" "{2}"'.format(inFile, inFile.replace('.tif', '_Binary.tif'), inFile.replace('.tif', '_labeled.pkl')))
    segmented = loadObjects(inFile)
    visualize_labeled(segmented, inFile, name='_segmented')
    for i in range(len(segmented)):
        for j in range(len(segmented[0])):
            if segmented[i,j] != 0:
                neighbor_points = getNeighborIndices(segmented, i, j)
                for npoint in neighbor_points:
                    #print(npoint)
                    if segmented[npoint[0], npoint[1]] != 0 and segmented[npoint[0], npoint[1]] != segmented[i,j]:
                        out_array[i,j] = 0
                        break
    skimage.external.tifffile.imsave(inFile.replace('.tif', '_BinaryPivots.tif'), out_array)
    return out_array, segmented

def loadObjects(inFile):
    with open(inFile.replace('.tif', '_segmented.pkl'), 'rb') as segmentedFile:
        segmented = pickle.load(segmentedFile, encoding='latin1')
    return segmented


def visualize_labeled(labeled, inFile, name='_labeled'):
    labeled_image = skimage.color.gray2rgb(labeled)
    colorLimit = skimage.dtype_limits(labeled, True)[1]
    colored = ([0, 0, colorLimit], [0, colorLimit, 0], [colorLimit, 0, 0], [colorLimit, colorLimit, 0], [colorLimit, 0, colorLimit], [0, colorLimit, colorLimit], [colorLimit, colorLimit, colorLimit])
    for i in range(len(labeled_image)):
        for j in range(len(labeled_image[0])):
            if labeled[i,j] == 0:
                labeled_image[i,j] = [0, 0, 0]
            else:
                labeled_image[i,j] = colored[labeled[i,j] % len(colored)]
    labeled_image = labeled_image.astype(np.uint8)
    #print(labeled_image.tolist()) 
    skimage.external.tifffile.imsave(inFile.replace('.tif', '{0}.tif'.format(name)), labeled_image)

def remove_largeNoise(labeled, num_objects):
    if num_objects == 0:
        return labeled
    for n in range(1, num_objects):
        obj = []
        for i in range(len(labeled)):
            for j in range(len(labeled[0])):
                if labeled[i][j] == n:
                    obj.append((i,j))
        if len(obj) > 0.1 * len(labeled)**2:
            for i,j in obj:
                labeled[i][j] = 0
    return labeled

def makeCells(inFile, clusters):

    labels_per_cluster = declumpKMeans(inFile, clusters, clusters[0].binary, Cluster.pic)
    assert len(clusters) == len(labels_per_cluster), 'each cluster must be labeled'
    for c, labels in zip(clusters, labels_per_cluster):
        c.splitConglomerates(labels)
    if Cluster.clusters:
        skimage.external.tifffile.imsave(inFile.replace('.tif', '_BinaryEdged.tif'), Cluster.clusters[0].binary)

def getBinary(inFile, pic_array, binarized):
    if binarized:
        try:
            with skimage.external.tifffile.TiffFile(inFile.replace('.tif', '_Binary.tif')) as pic_bin:
                bin_array = pic_bin.asarray()
            segmented = loadObjects(inFile)
        except (FileNotFoundError, EOFError):
            pic = skimage.external.tifffile.TiffFile(inFile)
            bin_array, segmented = makeBinary(inFile, pic_array, pic)
            pic.close()
        finally:
            global WHITE
            WHITE = skimage.dtype_limits(bin_array, True)[1]
            Binarize.WHITE = WHITE
            Binarize.bin_WHITE = WHITE
            Cell_objects.WHITE = WHITE
    else:
        pic = skimage.external.tifffile.TiffFile(inFile)
        bin_array, segmented = makeBinary(inFile, pic_array, pic)
        pic.close()
    # if virus_inject:
    #     vload = getLoadedPixels(pic_array, bin_array)
    #     skimage.external.tifffile.imsave(inFile.replace('.tif', '_virus.tif'), vload)
    return bin_array, segmented


def superimposeBoundary(inFile, pic_array, boundary=None):
    if boundary is None:
        boundary = []
        for c in Cluster.clusters:
            boundary.extend(c.boundary)
    bound = copy.deepcopy(pic_array)
    for b in boundary:
        bound[b[0]][b[1]] = 0
    skimage.external.tifffile.imsave(inFile.replace('.tif', '_Bound.tif'), bound)

def loadClusters(inFile, stack_slice):
    clustFile = open(inFile.replace('.tif', '_clusters.pkl'), 'rb') #FileNotFoundError if not found. DO NOT try-block!
    Cluster.clusters = pickle.load(clustFile)
    Cluster.segmented = pickle.load(clustFile)
    print('******', len(Cluster.clusters))
    for c in Cluster.clusters:
        #print("Cluster!!")
        c.stack_slice = stack_slice
    clustFile.close()
    return Cluster.clusters

def saveClusters(inFile, clusters=Cluster.clusters):
    outFile = open(inFile.replace('.tif', '_clusters.pkl'), 'wb')
    pickle.dump(clusters, outFile)
    pickle.dump(Cluster.segmented, outFile)
    outFile.close()

def loadCells(inFile, stack_slice, grid):
    cellFile = open(inFile.replace('.tif', '_cells.pkl'), 'rb') #FileNotFoundError if not found. DO NOT try-block!
    stack_slice.cells = pickle.load(cellFile)
    for c in stack_slice.cells:
        c.stack_slice = stack_slice
        c.gridSquare = grid.getGridSquare(c.center)
        c.gridSquare.addCell(c)
    cellFile.close()

def saveCells(inFile, stack_slice):
    #print("cells made")
    outFile = open(inFile.replace('.tif', '_cells.pkl'), 'wb')
    pickle.dump(stack_slice.cells, outFile)
    outFile.close()

def process_image(inFile, stack_slice, grid, binarized, clustered, split, overlay):
    with skimage.external.tifffile.TiffFile(inFile) as pic:
        pic_array = pic.asarray()
    global nucleusMode
    nucleusMode = '-rfp-' in inFile or 'ucleus' in inFile
    global virus_inject
    virus_inject = 'RFP' in inFile
    global WHITE
    if split: #breakpoint to test stack collation
        try:
            loadCells(inFile, stack_slice, grid)
            Cluster.pic = pic_array
        except (FileNotFoundError, EOFError):
            clustered = True
        else:
            WHITE = skimage.dtype_limits(pic_array, True)[1]
            Binarize.WHITE = WHITE
            Binarize.bin_WHITE = WHITE
            Cell_objects.WHITE = WHITE
            return pic_array
    if clustered:
        clusters = []
        try:
            clusters = loadClusters(inFile, stack_slice)
            Cluster.pic = pic_array
        except (FileNotFoundError, EOFError):
            binarized = True
            #return complete_protocol()
        else:
            WHITE = skimage.dtype_limits(pic_array, True)[1]
            Binarize.WHITE = WHITE
            Binarize.bin_WHITE = WHITE
            Cell_objects.WHITE = WHITE
            superimposeBoundary(inFile, pic_array)
            makeCells(inFile, clusters)
            saveCells(inFile, stack_slice)
            return pic_array

    else:
        pass

    bin_array, segmented = getBinary(inFile, pic_array, binarized)
    Cluster.pic = pic_array
    Cluster.segmented = segmented
    clusters, boundary = makeClusters_Matlab(bin_array, inFile, stack_slice)
    superimposeBoundary(inFile, pic_array)#, boundary=boundary)
    #visualize_Clusters(pic_array, inFile, Cluster.clusters)
    saveClusters(inFile, clusters)
    makeCells(inFile, clusters) #saves edged picture
    saveCells(inFile, stack_slice)
    return pic_array

def visualize_Clusters(out_array, inFile, clusters, num=None):    
    c_arr = copy.deepcopy(out_array) #psuedo-deep
    for i in range(len(c_arr)):
        for j in range(len(c_arr[0])):
            c_arr[i][j] = 0
    if num is None:        
        for k in range(len(clusters)):
            ck = [(c[0], c[1]) for c in clusters[k].boundary]
            for (i,j) in ck:
                c_arr[i][j] = WHITE
        skimage.external.tifffile.imsave(inFile.replace('.tif', '_clusters_' +'.tif'), c_arr)
    else:
        k = num
        ck = [(c[0], c[1]) for c in clusters[k].boundary]
        for i in range(len(c_arr)):
            for j in range(len(c_arr[0])):
                if (i,j) in ck:
                    c_arr[i][j] = WHITE
                else:
                    c_arr[i][j] = 0
        skimage.external.tifffile.imsave(inFile.replace('.tif', '_cluster_' +str(k)+'.tif'), c_arr) 

def parallel(prefix, binarized, clustered, split, overlaid):

    current_stack = Stack(512, 512, 32)
    Cluster.current_stack = current_stack
    x = 1
    if split:
        binarized, clustered = True, True
    elif clustered:
        binarized = True

    pic_arrays = []
    while True:
        try:
            stack_slice = Stack_slice(x, cells=[])
            inFile = prefix + 'piece-' + str(x).rjust(4, '0') + '.tif'

            pic_arrays.append(process_image(inFile, stack_slice, current_stack.grid, binarized, clustered, split, overlay))

            stack_slice.pruneCells(0.4)
            print("Slice #{0} has {1} cells : ".format(stack_slice.number, len(stack_slice.cells)))
            current_stack.addSlice(stack_slice)

        except IOError:
            if x != 1:
                break
            else:
                print("IOError")
                print('{0} not processed'.format(prefix))
                logging.error(traceback.format_exc())
                return
        else:
            print(prefix)
            x += 1


    current_stack.collate_slices(nucleusMode)
    overlay(current_stack, prefix, pic_arrays)


def groupPointsBoundary(boundary, num_groups=2):
    KMean = KMeans(n_clusters=num_groups)
    b = np.array(boundary)
    KMean.fit(b)#.reshape(-1,1))
    return list(KMean.labels_)

def groupPoints(points, num_groups, sample_weights=None):
    KMean = KMeans(n_clusters=num_groups)
    b = np.array(points)
    if sample_weights is not None:
        sw = np.array(sample_weights)
        try:
            KMean.fit(b, sw) #.reshape(-1,1))
        except ValueError:
            return [0 for k in range(len(points))]
    else:
        KMean.fit(b)
    return list(KMean.labels_)

def declumpKMeansBoundary(inFile, clusters):
    bin_array = clusters[0].binary
    out_rgb = skimage.color.gray2rgb(bin_array)
    out_rgb.fill(0)

    colorLimit = skimage.dtype_limits(out_rgb, True)[1]
    colored = ([0, colorLimit, 0], [colorLimit, 0, 0], [colorLimit, colorLimit, 0], [colorLimit, 0, colorLimit], [0, colorLimit, colorLimit], [colorLimit, colorLimit, colorLimit])

    labels_per_cluster = []

    for c in clusters:
        kmean_labels = groupPoints(c.boundary, max(1,len(c.pivots)+2))
        labels_per_cluster.append(kmean_labels)
        for p, k_label in zip(c.boundary, kmean_labels):
            out_rgb[p[0], p[1]] = colored[k_label % len(colored)]
    skimage.external.tifffile.imsave(inFile.replace('.tif', '_kmeans.tif'), out_rgb)
    return labels_per_cluster

def declumpKMeans(inFile, clusters, bin_array, pic_array):
    out_rgb = skimage.color.gray2rgb(bin_array)
    out_rgb.fill(0)
    colorLimit = skimage.dtype_limits(out_rgb, True)[1]
    colored = ([0, colorLimit, 0], [colorLimit, 0, 0], [colorLimit, colorLimit, 0], [colorLimit, 0, colorLimit], [0, colorLimit, colorLimit], [colorLimit, colorLimit, colorLimit])
    labels_per_cluster = []

    for c in clusters:
        _ = c.area #updates interior
        weights = []
        for ci in c.interior:
            weights.append(pic_array[ci[0], ci[1]]**2)
        kmean_labels = groupPoints(c.interior, num_groups=max(1,len(c.pivots)+1), sample_weights=weights)
        labels_per_cluster.append(kmean_labels)
        for p, k_label in zip(c.interior, kmean_labels):
            out_rgb[p[0], p[1]] = colored[k_label % len(colored)]
    skimage.external.tifffile.imsave(inFile.replace('.tif', '_kmeans.tif'), out_rgb)
    return labels_per_cluster


def overlay(current_stack, prefix, pic_arrays):

    out_rgb = skimage.color.gray2rgb(pic_arrays[0])
    out_rgb.fill(0)
    x = 0
    if nucleusMode:
        outFile = open(prefix + 'Nucleus Sizes.csv', 'w')
    else:
        outFile = open(prefix + 'Soma Sizes.csv', 'w')
    colorLimit = skimage.dtype_limits(out_rgb, True)[1]
    colored = ([0, 0, colorLimit], [0, colorLimit, 0], [colorLimit, 0, 0], [colorLimit, colorLimit, 0], [colorLimit, 0, colorLimit], [0, colorLimit, colorLimit], [colorLimit, colorLimit, colorLimit])
    cyan = [0, colorLimit, colorLimit]; magenta = [colorLimit, 0, colorLimit]; yellow = [colorLimit, colorLimit, 0]
    green = [0, colorLimit, 0]; white_ = [colorLimit, colorLimit, colorLimit]; red = [colorLimit, 0, 0]
    for c in current_stack.large_Cells:
        x+=1
        color_index = c.stack_slice.number % len(colored)
        for b in c.boundary:
            out_rgb[b[0]][b[1]] = colored[color_index]
    skimage.external.tifffile.imsave(prefix + 'largest.tif', out_rgb)
    outFile.write("LARGEST CELLS ARE \n")
    largest_3d = []
    x=1
    assert len(current_stack.stack_slices) == len(pic_arrays), 'Stack slices and picture array mismatches: {0} and {1}'.format(len(current_stack.stack_slices), len(pic_arrays))
    for ss, pic_array in zip(current_stack.stack_slices, pic_arrays):
        largest_3d.append(skimage.color.gray2rgb(pic_array))
        color_index = ss.number % len(colored)
        for c in ss.cells:
            for b in c.boundary:
                largest_3d[-1][b[0]][b[1]] = [0,0,0]
        for c in ss.overlapping_Cells: 
            for b in c.boundary:
                largest_3d[-1][b[0]][b[1]] = [ int(pic_array[b[0]][b[1]] * 0.5) + [int(c * 0.5) for c in white_][i] for i in range(0, 3)]
        for c in ss.roundness_rejected_Cells: #this is a subset of cells (above) so order matters!
            for b in c.boundary:
                largest_3d[-1][b[0]][b[1]] = [ int(pic_array[b[0]][b[1]] * 0.5) + [int(c * 0.5) for c in yellow][i] for i in range(0, 3)]
        for c in ss.size_rejected_Cells: 
            for b in c.boundary:
                largest_3d[-1][b[0]][b[1]] = [ int(pic_array[b[0]][b[1]] * 0.5) + [int(c * 0.5) for c in green][i] for i in range(0, 3)]
        for c in ss.contained_Cells: 
            for b in c.boundary:
                largest_3d[-1][b[0]][b[1]] = [ int(pic_array[b[0]][b[1]] * 0.5) + [int(c * 0.5) for c in magenta][i] for i in range(0, 3)]
        for c in ss.finalizedCellSlice.cells:
            outFile.write("Cell, {0}, Slice #, {1}, Area:, {2}, Roundness:, {3}, Viral status:, {4}\n".format(x, c.stack_slice.number, c.area, c.roundness, c.isLoaded))
            x+=1
            for b in c.interior:
                largest_3d[-1][b[0]][b[1]] = [ int(pic_array[b[0]][b[1]] * 0.8) + [int(c * 0.2) for c in cyan][i] for i in range(0, 3)]
            if c.isLoaded: 
                for b in c.boundary:
                    largest_3d[-1][b[0]][b[1]] = [ int(pic_array[b[0]][b[1]] * 0.5) + [int(c * 0.5) for c in red][i] for i in range(0, 3)]
            else:
                for b in c.boundary:
                    largest_3d[-1][b[0]][b[1]] = [ int(pic_array[b[0]][b[1]] * 0.5) + [int(c * 0.5) for c in green][i] for i in range(0, 3)]
        #skimage.external.tifffile.imsave(prefix + 'largest' + str(ss.number) + '.tif', largest_3d[-1])
    outFile.close()
    largest_3d = np.array(largest_3d)
    skimage.external.tifffile.imsave('{0}largest3D.tif'.format(prefix), largest_3d)

parser = argparse.ArgumentParser(description="specify previous completion")
parser.add_argument('-b', '--binarized', dest="binarized", default=False, action="store_true")
parser.add_argument('-c', '--clustered', dest="clustered", default=False, action="store_true")
parser.add_argument('-s', '--split', dest="split", default=False, action="store_true")
parser.add_argument('-o', '--overlaid', dest="overlaid", default=False, action="store_true")
parser.add_argument('-l', '--local', dest="localRun", default=False, action="store_true")
args = parser.parse_args()

def one_arg(prefix):
    try:
        parallel(prefix, args.binarized, args.clustered, args.split, args.overlaid)
    except Exception as e:
        print("Error occured in processing {0}: {1}".format(prefix, e))
        logging.error(traceback.format_exc())


def getImageDirectories(locations):
    prefixes = []

    def recursiveDirectories(loc):
        nonlocal prefixes
        try:
            for d in next(os.walk(loc))[1]:
                if 'normal' in d or '_RFP' in d:
                    prefixes.append(loc + d + '/')
                    print(loc + d + '/')
                else:
                    recursiveDirectories(loc + d + '/')
        except StopIteration:
            pass

    for loc in locations:
        recursiveDirectories(loc)
    return prefixes

if args.localRun:
    one_arg('../p2f1_normal/')
    #one_arg('../YFP-RA viruses imaging for morphology/3rd set of experiments/rd1-403/Mouse 1/LEFT EYE/cell-12-RFP/piece-')
    locations = [
    '../vit A/vit_A_free/',
    '../Cell-Size-Project/WT/',
    '../Cell-Size-Project/RD1-P2X7KO/',
    '../Cell-Size-Project/RD1/',
    '../VAF_new_cohort/',
    '../YFP-RA-viruses/'
    ]

else:
    locations = [
    '/global/scratch/arjitmisra/2-14/vit_A_free/',
    '/global/scratch/arjitmisra/2-14/WT/',
    '/global/scratch/arjitmisra/2-14/RD1-P2X7KO/',
    '/global/scratch/arjitmisra/2-14/RD1/',
    '/global/scratch/arjitmisra/2-14/VAF_new_cohort/',
    '/global/scratch/arjitmisra/YFP-RA-viruses/'
    ]
    prefixes = getImageDirectories(locations)
    cpus = multiprocessing.cpu_count()
    with Pool(cpus * 3) as p:
      p.map(one_arg, prefixes)

