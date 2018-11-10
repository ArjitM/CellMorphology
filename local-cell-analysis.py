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
import matlab.engine
from operator import add

def makeClusters_deprecated(binary, boundary, stack_slice):
    
    Cluster.clusters = []
    if len(boundary) == 0:
        print("NO BOUNDARY")
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
                #print(len(current), len(boundary))
                if len(current) > 1000:
                    print("STOPPP")
                    break
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
                #print("cluster extended")
        if len(current) > 12 and len(current) < 1000:
            clusterBounds.append(current)
            #print("###############################cluster appended")

    Cluster.clusters = []
    for c in clusterBounds:
        Cluster.clusters.append(Cluster(binary, c, stack_slice))
    print("$$$$$$$$$$$$$$", len(clusterBounds))
    return Cluster.clusters


def makeClusters(binary, inFile, stack_slice):
    eng = matlab.engine.start_matlab()
    img = eng.imread(inFile.replace('.tif', '_Binary.tif'))
    #print(img)
    boundaries = eng.bwboundaries(img)
    eng.quit()
    clusterBounds = []
    for bound in boundaries:
        clusterBounds.append([tuple( map( add, (np.array(bp).astype(np.uint16)), (-1, -1) ) ) for bp in bound]) #convert from matlab double
        #matlab array indexing starts at 1!
    boundary = [item for sublist in clusterBounds for item in sublist]
    for c in clusterBounds:
        Cluster.clusters.append(Cluster(binary, c, stack_slice))
    return Cluster.clusters, boundary

def makeBinary(inFile, pic_array, pic):

    #pic_array = pic.asarray()
    #out_array = pic.asarray(); #copy dimensions
    nucleusMode = '-rfp-' in inFile or 'ucleus' in inFile

    out_array = [ [0] * len(pic_array[1]) ] * len(pic_array)
    global WHITE
    WHITE = skimage.dtype_limits(pic_array, True)[1]
    #print("Limit: ", WHITE)
    Binarize.WHITE = WHITE
    Binarize.bin_WHITE = WHITE
    Cell_objects.WHITE = WHITE

    regions = Compartmentalize(pic_array, 32)

    #out_array = copy.deepcopy(pic_array) #
    out_array = np.array(out_array, dtype=pic_array.dtype.type) # keep same image type
    #basicEdge(pic_array, out_array, regions) # preliminary edge detection via pixel gradient

    out_array = pic_array > threshold_local(pic_array, block_size=35).astype(pic_array.dtype.type) #NOT SCALED MUST FIX
    out_array = out_array.astype(pic_array.dtype.type)
    out_array = np.array([ [WHITE if p else 0 for p in row] for row in out_array], dtype = pic_array.dtype.type)
    print(out_array)

    skimage.external.tifffile.imsave(inFile.replace('.tif', '_edgeBasic.tif'), out_array)

    regions.setNoiseCompartments(out_array, 0.95)

    #enhanceEdges(pic_array, out_array, regions, nucleusMode=False) # use detected averages to guess missing edges
    #skimage.external.tifffile.imsave(inFile.replace('.tif', '_edgeEnhance.tif'), out_array)

    noise_handler = Noise(out_array, iterations=3, binary=True)
    noise_handler.reduce() #reduce salt and pepper noise incorrectly labelled as edges 

    skimage.external.tifffile.imsave(inFile.replace('.tif', '_Binary.tif'), out_array)
    print("***made binary")
    return out_array


def makeCells(inFile, clusters=Cluster.clusters):
    noise_clusters = []
    i = -1
    for c in clusters:
        try:
            c.getTrueCusps(10)
        except AssertionError:
            noise_clusters.append(c)
        else:
            c.pruneCusps()
            c.propagateInternalBoundaries()
            #c.showCusps()  #WONT WORK with boolean binary
            c.splitByEdges()
    for c in noise_clusters:
        c.kill()
    if Cluster.clusters:
        skimage.external.tifffile.imsave(inFile.replace('.tif', '_BinaryEdged.tif'), Cluster.clusters[0].binary)


def getBinary(inFile, pic_array, binarized):
    if binarized:
        try:
            with skimage.external.tifffile.TiffFile(inFile.replace('.tif', '_Binary.tif')) as pic_bin:
                bin_array = pic_bin.asarray()
        except (FileNotFoundError, EOFError):
            pic = skimage.external.tifffile.TiffFile(inFile)
            bin_array = makeBinary(inFile, pic_array, pic)
    else:
        pic = skimage.external.tifffile.TiffFile(inFile)
        bin_array = makeBinary(inFile, pic_array, pic)
    pic.close()
    return bin_array


def superimposeBoundary(inFile, pic_array, boundary):
    bound = copy.deepcopy(pic_array)
    for b in boundary:
        bound[b[0]][b[1]] = 0
    skimage.external.tifffile.imsave(inFile.replace('.tif', '_Bound.tif'), bound)

def loadClusters(inFile, stack_slice):
    clustFile = open(inFile.replace('.tif', '_clusters.pkl'), 'rb') #FileNotFoundError if not found. DO NOT try-block!
    Cluster.clusters = pickle.load(clustFile)
    #print(len(Cluster.clusters))
    for c in Cluster.clusters:
        #print("Cluster!!")
        c.stack_slice = stack_slice
    clustFile.close()
    return Cluster.clusters

def saveClusters(inFile, clusters=Cluster.clusters):
    #print("clusters made")
    outFile = open(inFile.replace('.tif', '_clusters.pkl'), 'wb')
    #print("will be saved ", len(clusters))
    pickle.dump(clusters, outFile)
    outFile.close()

def loadCells(inFile, stack_slice):
    cellFile = open(inFile.replace('.tif', '_cells.pkl'), 'rb') #FileNotFoundError if not found. DO NOT try-block!
    stack_slice.cells = pickle.load(cellFile)
    for c in stack_slice.cells:
        c.stack_slice = stack_slice
    cellFile.close()

def saveCells(inFile, stack_slice):
    #print("cells made")
    outFile = open(inFile.replace('.tif', '_cells.pkl'), 'wb')
    pickle.dump(stack_slice.cells, outFile)
    outFile.close()


def process_image(inFile, stack_slice, binarized, clustered, split, overlay):

    with skimage.external.tifffile.TiffFile(inFile) as pic:
        pic_array = pic.asarray()

    if split: #breakpoint to test stack collation
        try:
            loadCells(inFile, stack_slice)
        except (FileNotFoundError, EOFError):
            clustered = True
        else:
            return pic_array

    if clustered:
        clusters = []
        try:
            clusters = loadClusters(inFile, stack_slice)
        except (FileNotFoundError, EOFError):
            bin_array = getBinary(inFile, pic_array, binarized=True)
            #boundary = findBoundaryPoints(bin_array)
            Cluster.pic = pic_array
            clusters, boundary = makeClusters(bin_array, inFile, stack_slice)
            superimposeBoundary(inFile, pic_array, boundary)
            saveClusters(inFile, clusters)
        finally:
            makeCells(inFile, clusters)
            saveCells(inFile, stack_slice)
            return pic_array

    else:
        bin_array = getBinary(inFile, pic_array, binarized)
        #print(bin_array)
        #print("#######hhhhhhhhhh", len(bin_array))
        #boundary = findBoundaryPoints(bin_array)
        #print("#############", len(boundary))
        
        Cluster.pic = pic_array
        clusters = makeClusters(bin_array, inFile, stack_slice)[0]
        saveClusters(inFile, clusters)  
        makeCells(inFile, clusters) 
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


locations = [
'../vit A/vit_A_free/',
'../Cell Size Project/WT/',
'../Cell Size Project/RD1-P2X7KO/',
'../Cell Size Project/RD1/'
]

prefixes = []
for loc in locations:
    for dir1 in next(os.walk(loc))[1]: #Expt #/Mouse#
        try:
            for dir2 in next(os.walk(loc + dir1))[1]:
                if 'normal' in dir2:
                    prefixes.append(loc + dir1 + "/" + dir2 + "/piece-")
                else:
                    for dir3 in next(os.walk(loc + dir1 + "/" + dir2))[1]:
                        if 'normal' in dir3:
                            prefixes.append(loc + dir1 + "/" + dir2 + "/" + dir3 + "/piece-")
        except:
            pass


def parallel(prefix, binarized, clustered, split, overlaid):

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
            try:
                subprocess.run('convert {0} {1}'.format(inFile.replace(' ', "\\ ").replace('.tif', '.jpg'), inFile.replace(' ', "\\ ")), shell=True)
            except FileNotFoundError:
                pass 

            pic_arrays.append(process_image(inFile, stack_slice, binarized, clustered, split, overlay))

            stack_slice.pruneCells(0.65)
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

    current_stack.collate_slices()
    overlay(current_stack, prefix, pic_arrays)

def overlay(current_stack, prefix, pic_arrays):

    out_rgb = skimage.color.gray2rgb(pic_arrays[0])
    out_rgb.fill(0)
    x = 0
    if "-rfp-" in prefix:
        outFile = open(prefix + 'Nucleus Sizes.csv', 'w')
    else:
        outFile = open(prefix + 'Soma Sizes.csv', 'w')

    colorLimit = skimage.dtype_limits(out_rgb, True)[1]
    colored = ([0, 0, colorLimit], [0, colorLimit, 0], [colorLimit, 0, 0], [colorLimit, colorLimit, 0], [colorLimit, 0, colorLimit], [0, colorLimit, colorLimit], [colorLimit, colorLimit, colorLimit])
    cyan = [0, colorLimit, colorLimit]; magenta = [colorLimit, 0, colorLimit]

    outFile.write("LARGEST CELLS ARE \n")
    for c in current_stack.large_Cells:
        x+=1
        color_index = c.stack_slice.number % len(colored)
        for b in c.boundary:
            out_rgb[b[0]][b[1]] = colored[color_index]
    skimage.external.tifffile.imsave(prefix + 'largest.tif', out_rgb)
    
    largest_3d = []
    for ss, pic_array in zip(current_stack.stack_slices, pic_arrays):
        largest_3d.append(skimage.color.gray2rgb(pic_array))
        #largest_3d[-1].fill(0)
        color_index = ss.number % len(colored)

        #print(ss.finalizedCellSlice.cells)
        for c in ss.cells:
            for b in c.boundary:
                largest_3d[-1][b[0]][b[1]] = magenta

        for c in ss.finalizedCellSlice.cells:
            outFile.write("Cell, {0}, Slice #, {1}, Area:, {2}, Roundness:, {3}\n".format(x, c.stack_slice.number, c.area, c.roundness))
            for b in c.interior:
                largest_3d[-1][b[0]][b[1]] = [ int(pic_array[b[0]][b[1]] * 0.8) + [int(c * 0.2) for c in cyan][i] for i in range(0, 3)]
            for b in c.boundary:
                largest_3d[-1][b[0]][b[1]] = cyan
                #colored[color_index]
        #skimage.external.tifffile.imsave(prefix + 'largest' + str(ss.number) + '.tif', largest_3d[-1])
    outFile.close()
    largest_3d = np.array(largest_3d)
    skimage.external.tifffile.imsave('{0}largest3D.tif'.format(prefix), largest_3d)



parser = argparse.ArgumentParser(description="specify previous completion")

parser.add_argument('-b', '--binarized', dest="binarized", default=False, action="store_true")
parser.add_argument('-c', '--clustered', dest="clustered", default=False, action="store_true")
parser.add_argument('-s', '--split', dest="split", default=False, action="store_true")
parser.add_argument('-o', '--overlaid', dest="overlaid", default=False, action="store_true")

args = parser.parse_args()

def one_arg(prefix):
    try:
        parallel(prefix, args.binarized, args.clustered, args.split, args.overlaid)
    except Exception as e:
        print("Error occured in processing {0}: {1}".format(prefix, e))
        logging.error(traceback.format_exc())
    # try:
    #     subprocess.run("rclone move {0} arjit_bdrive:/Cell_Morphology_Research/{0}".format(prefix.replace("/piece-", "")), shell=True)
    # except Exception as e:
    #     print("Error occured in copying {0}: {1}".format(prefix, e))
    #     logging.error(traceback.format_exc())

# cpus = multiprocessing.cpu_count()
# with Pool(1) as p:
#   p.map(one_arg, prefixes[:4])

one_arg('../Cell Size Project/RD1/expt_1/piece2-gfp-normal/piece-')






