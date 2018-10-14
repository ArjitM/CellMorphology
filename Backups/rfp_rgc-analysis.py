import multiprocessing
from multiprocessing import Pool

import Binarize
import Cell_objects
from Binarize import *
from Cell_objects import *
from Stack_objects import *
import numpy as np
from skimage import util
import subprocess
import copy
import argparse
import pickle

def makeClusters(binary, boundary, stack_slice):
    
    Cluster.clusters = []
    if len(boundary) == 0:
        return []
    boundary.sort() #should be sorted but double-checking; sort by i then j
    clusterBounds = []
    while boundary:
        #clusterBounds.append([boundary[0]])
        current = [boundary[0]]
        pivot = boundary.pop(0)
        #current = clusterBounds[-1] #last cluster element
        while True:
            if current:
                point = current[-1]
            else:
                break
            neighbor_points = filter(lambda x: (x[0], x[1]) in getNeighborIndices(binary, point[0], point[1]), boundary)
            try:
                neighbor = next(neighbor_points)
            except StopIteration:    
                neighbors = getNeighborIndices(binary, point[0], point[1])

                if (pivot[0], pivot[1]) not in neighbors:
                    #remove internal loops if present; switch control back to point where inner loop intersects boundary and delete loop points
                    k = -2 #start with second-last point
                    internalLoop = False
                    while abs(k) <= len(current):
                        if (current[k][0], current[k][1]) in neighbors:
                            del current[k+1:]
                            internalLoop = True
                            break
                        k -= 1

                    if not internalLoop: #could be extended line
                        ex_line = False
                        k = -2 #start with second-last point
                        while abs(k) <= len(current):
                            fork = iter(getNeighborIndices(binary, current[k][0], current[k][1]))
                            for n in fork: #getNeighborIndices(binary, current[k][0], current[k][1]):
                                if n in boundary:
                                    del current[k+1:]
                                    current.append(n)
                                    ex_line = True #breaking out of while loop
                                    break
                            if ex_line:
                                break
                            k -= 1

                        if not ex_line: #not internal loop nor extended line
                            #print("cluster boundary incomplete")
                            current.pop()
                else:
                    break #cluster is contiguous; exits outer while loop
            else:
                current.append(neighbor)
                boundary.remove(neighbor)
        if len(current) > 12 and len(current) < 1000:
            clusterBounds.append(current)

    clusters = []
    for c in clusterBounds:
        clusters.append(Cluster(binary, c, stack_slice))
    print("$$$$$$$$$$$$$$", len(clusterBounds))
    return clusters

def makeBinary(inFile, pic_array):

    #pic_array = pic.asarray()
    #out_array = pic.asarray(); #copy dimensions
    out_array = [ [0] * len(pic_array[1]) ] * len(pic_array)
    global WHITE
    WHITE = skimage.dtype_limits(pic_array, True)[1]
    Binarize.WHITE = WHITE
    Cell_objects.WHITE = WHITE

    regions = Compartmentalize(pic_array, 32)

    basicEdge(pic_array, out_array, regions) # preliminary edge detection via pixel gradient
    out_array = np.array(out_array, dtype=np.int8) # CONVERT TO boolean binary image
    skimage.external.tifffile.imsave(inFile.replace('.tif', '_edgeBasic.tif'), out_array)

    regions.setNoiseCompartments(out_array, 0.95)

    enhanceEdges(pic_array, out_array, regions) # use detected averages to guess missing edges
    skimage.external.tifffile.imsave(inFile.replace('.tif', '_edgeEnhance.tif'), out_array)

    noise_handler = Noise(out_array, iterations=5, binary=True)
    noise_handler.reduce() #reduce salt and pepper noise incorrectly labelled as edges 

    if '-rfp-' in inFile:
        out_array = skimage.util.invert(out_array) #inversion required for rfp labelled cells (code originally written for gfp)
        
    skimage.external.tifffile.imsave(inFile.replace('.tif', '_Binary.tif'), out_array)
    return out_array


def makeCells(clusters):
    noise_clusters = []
    i = -1
    for c in clusters:
        try:
            c.getTrueCusps(10)
        except AssertionError:
            noise_clusters.append(c)
        finally:
            c.pruneCusps()
            c.propagateInternalBoundaries()
            #c.showCusps()  #WONT WORK with boolean binary
            c.splitByEdges()
    for c in noise_clusters:
        c.kill()

def getBinary(inFile, pic_array, binarized):
    if binarized:
        try:
            with skimage.external.tifffile.TiffFile(inFile.replace('.tif', '_Binary.tif')) as pic_bin:
                bin_array = pic_bin.asarray()
        except FileNotFoundError:
            pass
        else:
            bin_array = makeBinary(inFile, pic_array)
    else:
        bin_array = makeBinary(inFile, pic_array)
    print("***made binary")
    return bin_array


def superimposeBoundary(inFile, pic_array, boundary):
    bound = copy.deepcopy(pic_array)
    for b in boundary:
        bound[b[0]][b[1]] = 0
    skimage.external.tifffile.imsave(inFile.replace('.tif', '_Bound.tif'), bound)

def loadClusters(inFile):
    clustFile = open(inFile.replace('.tif', '_clusters.pkl'), 'rb') #FileNotFoundError if not found. DO NOT try-block!
    Cluster.clusters = pickle.load(cellFile)
    clustFile.close()

def saveClusters(inFile, clusters=Cluster.clusters):
    outFile = open(inFile.replace('.tif', '_clusters.pkl'), 'wb')
    pickle.dump(clusters, outFile)
    outFile.close()

def loadCells(inFile, stack_slice):
    cellFile = open(inFile.replace('.tif', '_clusters.pkl'), 'rb') #FileNotFoundError if not found. DO NOT try-block!
    stack_slice.cells = pickle.load(cellFile)
    cellFile.close()

def saveCells(inFile, stack_slice):
    outFile = open(inFile.replace('.tif', '_clusters.pkl'), 'wb')
    print(stack_slice)
    pickle.dump(stack_slice.cells, outFile)
    outFile.close()


def process_image(inFile, stack_slice, binarized, clustered, split):

    with skimage.external.tifffile.TiffFile(inFile) as pic:
        pic_array = pic.asarray()

    if split: #breakpoint to test stack collation
        try:
            loadCells()
        except FileNotFoundError:
            clustered = True
        else:
            return pic_array

    if clustered:
        clusters = []
        try:
            clusters = loadClusters()
        except FileNotFoundError:
            bin_array = getBinary(inFile, pic_array, binarized=True)
            boundary = findBoundaryPoints(bin_array)
            Cluster.pic = pic_array
            clusters = makeClusters(bin_array, boundary, stack_slice)
            saveClusters(inFile)
        finally:
            makeCells(clusters)
            saveCells(inFile, stack_slice)
            return pic_array

    else:
        bin_array = getBinary(inFile, pic_array, binarized)
        boundary = findBoundaryPoints(bin_array)
        Cluster.pic = pic_array
        clusters = makeClusters(bin_array, boundary, stack_slice)
        saveClusters(inFile)  
        makeCells(clusters) 
        saveCells(inFile, stack_slice)
        return pic_array

   # test = internalBorderTest(pic_array, out_array, boundary)
    #visualize_Clusters(clusters, out_array, inFile)

    skimage.external.tifffile.imsave(inFile.replace('.tif', '_BinaryEdged.tif'), out_array)
    return pic_array


def visualize_Clusters(clusters, out_array, inFile):
    print(len(clusters))
    for c in clusters:
        print(c)
    
    c_arr = out_array[:]
    for i in range(len(c_arr)):
        for j in range(len(c_arr[0])):
            c_arr[i][j] = 0
            
    for k in range(len(clusters)):
        ck = [(c[0], c[1]) for c in clusters[k].boundary]

        
        for i in range(len(c_arr)):
            for j in range(len(c_arr[0])):
                if (i,j) in ck:
                    c_arr[i][j] = WHITE
                else:
                    c_arr[i][j] = 0
        skimage.external.tifffile.imsave(inFile.replace('.tif', '_cluster_' +str(k)+'.tif'), c_arr)


prefixes = [
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/RD1/piece1-rfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/RD1/piece2-rfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/RD1/piece3-rfp-normal/piece-',

'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/WT/piece1-rfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/WT/piece2-rfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/WT/piece3-rfp-normal/piece-',

'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/RD1/piece1-rfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/RD1/piece2-rfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/RD1/piece3-rfp-normal/piece-',

'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/WT/piece1-rfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/WT/piece2-rfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/WT/piece3-rfp-normal/piece-',

'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/RD1/piece1-rfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/RD1/piece2-rfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/RD1/piece3-rfp-normal/piece-',

'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/WT/piece1-rfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/WT/piece2-rfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/WT/piece3-rfp-normal/piece-']

def parallel(prefix, binarized, clustered, split):

    current_stack = Stack()
    x = 1
    if split:
        binarized, clustered = True, True
    elif clustered:
        binarized = True

    pic_arrays = []
    while True:
        try:
            stack_slice = Stack_slice(x, cells=[])
            inFile = prefix + str(x).rjust(4, '0') + '.tif'
            print("************",inFile)
            try:
                subprocess.run('convert {0} {1}'.format(inFile.replace(' ', "\\ ").replace('.tif', '.jpg'), inFile.replace(' ', "\\ ")))
            except FileNotFoundError:
                pass 

            pic_arrays.append(process_image(inFile, stack_slice, binarized, clustered, split))

            stack_slice.pruneCells(0.75)
            print("Slice #{0} has {1} cells : ".format(stack_slice.number, len(stack_slice.cells)))
            current_stack.addSlice(stack_slice)

        except IOError:
            print("IOError")
            if x != 0:
                break
            else:
                print('{0} not processed'.format(prefix))
                return
        else:
            print(prefix)
            x += 1
    current_stack.collate_slices()

    out_rgb = skimage.color.gray2rgb(pic_arrays[0])
    out_rgb.fill(0)
    x = 0

    outFile = open(prefix + 'Nucleus Sizes.csv', 'w')

    colorLimit = 255
    colored = ([0, 0, colorLimit], [0, colorLimit, 0], [colorLimit, 0, 0], [colorLimit, colorLimit, 0], [colorLimit, 0, colorLimit], [0, colorLimit, colorLimit], [colorLimit, colorLimit, colorLimit])

    outFile.write("LARGEST CELLS ARE \n")
    for c in current_stack.large_Cells:
        x+=1
        outFile.write("Cell, {0}, Slice #, {1}, Area:, {2}, Roundness:, {3}\n".format(x, c.stack_slice.number, c.area, c.roundness))
        color_index = c.stack_slice.number % len(colored)
        for b in c.boundary:
            out_rgb[b[0]][b[1]] = colored[color_index]

    outFile.close()
    skimage.external.tifffile.imsave(prefix + 'largest.tif', out_rgb)

    largest_3d = []

    for ss, pic_array in zip(current_stack.stack_slices, pic_arrays):
        largest_3d.append(skimage.color.gray2rgb(pic_arrays))
        largest_3d[-1].fill(0)
        color_index = ss.number % len(colored)

        for c in ss.finalizedCellSlice.cells:
            for b in c.boundary:
                largest_3d[-1][b[0]][b[1]] = colored[color_index]
        #skimage.external.tifffile.imsave(prefix + 'largest' + str(ss.number) + '.tif', largest_3d[-1])
    largest_3d = np.array(largest_3d)
    skimage.external.tifffile.imsave('{0}largest3D.tif'.format(prefix), largest_3d)


parser = argparse.ArgumentParser(description="specify previous completion")

parser.add_argument('-b', '--binarized', dest="binarized", default=False, action="store_true")
parser.add_argument('-c', '--clustered', dest="clustered", default=False, action="store_true")
parser.add_argument('-s', '--split', dest="split", default=False, action="store_true")

args = parser.parse_args()

def one_arg(prefix):
    parallel(prefix, args.binarized, args.clustered, args.split)


test_prefixes = [
'/Users/arjitmisra/Documents/Kramer Lab/timetest/eye1p1f2_normal/eye1-',
'/Users/arjitmisra/Documents/Kramer Lab/timetest/eye1p1f3_normal/eye1-'
]

cpus = multiprocessing.cpu_count()
with Pool(2) as p:
  p.map(one_arg, test_prefixes)

# for p in prefixes:
#     try:
#         parallel(p)
#     except:
#         print('\n\n{0} WAS NOT PROCESSED\n\n'.format(p))

#parallel('/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/RD1/piece1-rfp-normal/piece-')




