from multiprocessing import Pool

from Binarize import *
from Cell_objects import *
from Stack_objects import *


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
        if len(current) > 12:
            clusterBounds.append(current)

    clusters = []
    for c in clusterBounds:
        clusters.append(Cluster(binary, c, stack_slice))
    print("$$$$$$$$$$$$$$", len(clusterBounds))
    return clusters


def process_image(inFile, stack_slice):

    with skimage.external.tifffile.TiffFile(inFile) as pic:

        pic_array = pic.asarray();
        out_array = pic.asarray(); #copy dimensions
        global WHITE
        WHITE = skimage.dtype_limits(pic_array, True)[1]
        regions = Compartmentalize(pic_array, 32)

        basicEdge(pic_array, out_array, regions) # out_array is modified
        skimage.external.tifffile.imsave(inFile.replace('.tif', '_edgeBasic.tif'), out_array)
        #skimage.external.tifffile.imsave(inFile.replace('.tif', '_edgeSobel.tif'), skimage.img_as_uint(skimage.filters.sobel(pic_array)))

        regions.setNoiseCompartments(out_array, 0.95)

        enhanceEdges(pic_array, out_array, regions)
        skimage.external.tifffile.imsave(inFile.replace('.tif', '_edgeEnhance.tif'), out_array)

        noise_handler = Noise(out_array, iterations=3, binary=True)
        noise_handler.reduce()

        skimage.external.tifffile.imsave(inFile.replace('.tif', '_Binary.tif'), out_array)

        print("***made binary")

        boundary = findBoundaryPoints(out_array)

        bound = pic.asarray()
        for b in boundary:
            bound[b[0]][b[1]] = 0

       # test = internalBorderTest(pic_array, out_array, boundary)
        skimage.external.tifffile.imsave(inFile.replace('.tif', '_Bound.tif'), bound)

        Cluster.pic = pic_array
        clusters = makeClusters(out_array, boundary, stack_slice)
        noise_clusters = []
        i = -1
        for c in clusters:
            i += 1
            try:
                #c.showCusps(7)
                c.getTrueCusps(9)
            except AssertionError:
                noise_clusters.append(c)
            finally:
                c.pruneCusps()
                c.propagateInternalBoundaries()
                c.showCusps()
                c.splitByEdges()

        for c in noise_clusters:
            c.kill()

        skimage.external.tifffile.imsave(inFile.replace('.tif', '_BinaryEdged.tif'), out_array)
        return out_array


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
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p1f2_normal/eye1-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p1f3_normal/eye1-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p2f1_normal/eye1-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p2f2_normal/eye1-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p2f3_normal/eye1-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye2p1f1_normal/eye2-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye2p1f2_normal/eye2-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye2p1f3_normal/eye2-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p1f1_normal/eye1-',

# '/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/RD1/piece1-gfp-normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/RD1/piece2-gfp-normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/RD1/piece3-gfp-normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/RD1-P2X7KO/piece1-gfp-normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/RD1-P2X7KO/piece2-gfp-normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/RD1-P2X7KO/piece3-gfp-normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/WT/piece1-gfp-normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/WT/piece2-gfp-normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/WT/piece3-gfp-normal/piece-',

# '/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/RD1/piece1-gfp-normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/RD1/piece2-gfp-normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/RD1/piece3-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/RD1-P2X7KO/piece1-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/RD1-P2X7KO/piece2-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/RD1-P2X7KO/piece3-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/WT/piece1-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/WT/piece2-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 2/WT/piece3-gfp-normal/piece-',

'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/RD1/piece1-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/RD1/piece2-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/RD1/piece3-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/RD1-P2X7KO/piece1-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/RD1-P2X7KO/piece2-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/RD1-P2X7KO/piece3-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/WT/piece1-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/WT/piece2-gfp-normal/piece-',
'/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 3/WT/piece3-gfp-normal/piece-' #,

# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 2/eye1p1f1_normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 2/eye1p1f2_normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 2/eye1p1f3_normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 2/eye1p2f1_normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 2/eye1p2f2_normal/piece-',

# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 3/eye1p1f1_normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 3/eye1p1f2_normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 3/eye1p1f3_normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 3/eye1p2f1_normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 3/eye1p2f2_normal/piece-',
# '/Users/arjitmisra/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 3/eye1p2f3_normal/piece-',
]

def parallel(prefix):

    current_stack = Stack()
    x = 1
    out_array = []
    while True:
        try:
            stack_slice = Stack_slice(x, cells=[])
            out_array = process_image(prefix + str(x).rjust(4, '0') + '.tif', stack_slice)
            stack_slice.pruneCells()
            print("Slice #{0} has {1} cells : ".format(stack_slice.number, len(stack_slice.cells)))
            current_stack.addSlice(stack_slice)

        except IOError:
            break
        else:
            print(prefix)
            x += 1
    current_stack.collate_slices()

    out_rgb = skimage.color.gray2rgb(out_array)
    out_rgb.fill(0)
    x = 0

    outFile = open(prefix + 'Cell Sizes.csv', 'w')

    colored = ([0, 0, WHITE], [0, WHITE, 0], [WHITE, 0, 0], [WHITE, WHITE, 0], [WHITE, 0, WHITE], [0, WHITE, WHITE], [WHITE, WHITE, WHITE])

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

    for ss in current_stack.stack_slices:
        largest_3d.append(skimage.color.gray2rgb(out_array))
        largest_3d[-1].fill(0)
        color_index = ss.number % len(colored)

        for c in ss.finalizedCellSlice.cells:

            for b in c.boundary:

                largest_3d[-1][b[0]][b[1]] = colored[color_index]

        skimage.external.tifffile.imsave(prefix + 'largest' + str(ss.number) + '.tif', largest_3d[-1])

    largest_3d = np.array(largest_3d)
    skimage.external.tifffile.imsave('{0}largest3D.tif'.format(prefix), largest_3d)



#with Pool(2) as p:
#   p.map(parallel, prefixes)

# for p in prefixes:
#     try:
#         parallel(p)
#     except:
#         print('\n\n{0} WAS NOT PROCESSED\n\n'.format(p))

parallel('/Users/arjitmisra/Documents/Kramer Lab/Cell Size Project/experiment 1/RD1-P2X7KO/piece1-gfp-normal/piece-')




