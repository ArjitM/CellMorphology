from multiprocessing import Pool
import numpy as np 
#from __future__ import division

import skimage
from skimage import io
from skimage import external
from skimage import morphology
from skimage import filters
import math



class Compartment:
    def __init__(self, left, right, top, bottom, image_values):
        self.values = []
        for i in range(top, bottom):
            for j in range(left, right):
                self.values.append(float(image_values[i][j]))
        self.coordinates = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
        self.noise_compartment = False
        self.hasBorder = True
        self.neighbors = None

    def __str__(self):
        return self.coordinates

    def std(self):
        if self.values == []:
            print(self.coordinates)
            return 5
        return np.std(self.values, dtype=np.float64)

    def mean(self):
        return np.mean(self.values)

    def borderPercent(self, binary):
        border = 0
        for i in range(self.coordinates['top'], self.coordinates['bottom']):
            for j in range(self.coordinates['left'], self.coordinates['right']):
                if binary[i][j] == 0:
                    border += 1
        area = (self.coordinates['right'] - self.coordinates['left']) * (self.coordinates['bottom'] - self.coordinates['top'])
        return border / area


    def set_noise_compartment(self, value):
        self.noise_compartment = bool(value)

    def avgBorder(self, binary):
        k, border = 0, []
        for i in range(self.coordinates['top'], self.coordinates['bottom']):
            for j in range(self.coordinates['left'], self.coordinates['right']):
                if binary[i][j] == 0:
                    border.append(self.values[k])
                k += 1
        if border != []:
            return np.mean(border)

        self.hasBorder = False
        print(self.str())
        return None


class Compartmentalize:
    def __init__(self, image_values, size):
        self.compartments = []
        i_max, j_max, ki, kj = len(image_values), len(image_values[0]), 0, 0

        while ki < i_max:
            while kj < j_max:

                self.compartments.append(Compartment(kj, min(kj+size, j_max), ki, min(ki+size, i_max), image_values))
                kj += size
            kj = 0
            ki+= size
        self.size = size
        self.i_max = i_max
        self.j_max = j_max

    def getCompartment(self, i, j):
        index = (j // self.size) + (i // self.size) * (self.j_max // self.size)
        assert i >= 0 and j>= 0 and index < len(self.compartments)
        return self.compartments[index]

    def setNeighborCompartments(self, compartment):
        neighbors = []
        neighbor_points = [(compartment.coordinates['top'], compartment.coordinates['left'] - 1),
        (compartment.coordinates['top'] - 1, compartment.coordinates['left'] - 1),
        (compartment.coordinates['top'] - 1, compartment.coordinates['left']),
        (compartment.coordinates['bottom'], compartment.coordinates['right'] + 1),
        (compartment.coordinates['bottom'] + 1, compartment.coordinates['right'] + 1),
        (compartment.coordinates['bottom'] + 1, compartment.coordinates['right'])]
        
        for np in neighbor_points:
            try:
                neighbors.append(self.getCompartment(np[0], np[1]))
            except AssertionError:
                pass
        compartment.neighbors = neighbors

    def setNoiseCompartments(self, binary, border_thresh):
        for c in self.compartments:
            k = c.borderPercent(binary)
            if k > border_thresh:
                c.set_noise_compartment(True)
            else:
                c.set_noise_compartment(False)
        self.reviseNoiseCompartments()

    def reviseNoiseCompartments(self):
        for c in self.compartments:
            self.setNeighborCompartments(c)
        for c in self.compartments:
            hits = 0
            for n in c.neighbors:
                if n.noise_compartment:
                    hits += 1
            if hits > len(c.neighbors) / 2:
                c.set_noise_compartment(True)

    def getAvgBorder(self, binary):
        not_Noise = filter(lambda c: not bool(c.noise_compartment), self.compartments)      
        return np.mean([c.avgBorder(binary) for c in not_Noise if c.hasBorder])


class Noise:

    def __init__(self, image_values, regions=None, iterations=3, binary=False):
        if not binary:
            assert isinstance(regions, Compartmentalize), "Compartments are required for noise editing unless binary image"
            self.regions = regions
        self.image_values = image_values
        self.iterations = iterations
        self.binary = binary
        self.detectNoise = self.detectNoise_BIN if binary else self.detectNoise_OG

    def detectNoise_OG(self, i, j, neighbors):
        cut = regions.getCompartment(i, j).std() / 3
        num = len(neighbors)
        hits = 0
        for (y, x) in neighbors:
            if abs(float(self.image_values[i][j]) - float(self.image_values[y][x])) > cut:
                hits += 1
        return [hits > (num * 0.5), 0]

    def detectNoise_BIN(self, i, j, neighbors):
        num = len(neighbors)
        hits = 0
        hit_vals = []
        for (y, x) in neighbors:
            if self.image_values[i][j] != self.image_values[y][x]:
                hits += 1
                hit_vals.append(1)
            else:
                hit_vals.append(0)
        return [hits > (num * 0.5), WHITE] if self.image_values[i][j] == 0 else [hits > (num * 0.5), 0]

    def smooth(self):
        for i in range(len(self.image_values)):
            for j in range(len(self.image_values)):
                change, value = self.detectNoise(i, j, getNeighborIndices(self.image_values, i, j))
                if change:
                    self.image_values[i][j] = value

    def reduce(self):
        for _ in range(self.iterations):
            self.smooth()


def getNeighborIndices(image_values, i, j):
    neighbors = []
    for ki in range(-1, 2):
        for kj in range(-1, 2):
            if i + ki >= 0 and j + kj >= 0 and i + ki < len(image_values) and j + kj < len(image_values[0]) and not (ki==0 and kj==0):
                neighbors.append((i+ki, j+kj))
    return neighbors

def getForwardNeighbors(image, previous, point):
    assert previous != point, "Previous point cannot be the same point"
    delta_i = point[0] - previous[0]
    delta_j = point[1] - previous[1]
    not_diagonal = abs(delta_i) ^ abs(delta_j)
    if abs(not_diagonal) >= 2:
        print("Previous ", previous, " | Point ", point, " | Diagonal ", not_diagonal)
        raise AssertionError("previous point must be neighbor, diagonal: ", not_diagonal)
    
    if not_diagonal:
        if delta_i:
            possible = [(point[0] + delta_i, point[1] + k) for k in [-1, 0, 1]]
        else:
            possible = [(point[0] + k, point[1] + delta_j) for k in [-1, 0, 1]]
    else:
        possible = [(point[0] + delta_i, point[1]), (point[0] + delta_i, point[1] + delta_j), (point[0], point[1] + delta_j)]
        
    return [p for p in possible if p[0] < len(image) and p[1] < len(image[0])]
    



def basicEdge(pic_array, out_array, regions):
    for i in range(len(pic_array) - 1):
        for j in range(len(pic_array[0]) - 1):

            left, right = float(pic_array[i][j]), float(pic_array[i][j+1])
            bottom, diagonal = float(pic_array[i+1][j]), float(pic_array[i+1][j+1])

            ''' May require revision '''
            cut = regions.getCompartment(i, j).std() / 3  # 3 basic, 5 sensitive

            if abs(left - right) > cut:
                out_array[i][j] = 0
            elif abs(left - bottom) > cut:
                out_array[i][j] = 0
            elif abs(left - diagonal) > cut:
                out_array[i][j] = 0
            elif left ==  WHITE:
                out_array[i][j] = 0 #remove saturated pixels
            else:
                out_array[i][j] = WHITE

def enhanceEdges(pic_array, out_array, regions):
    global borderMean
    borderMean = regions.getAvgBorder(out_array)
    tolerance = 0.07 * WHITE # +/- 7 % of total image range
    for i in range(len(out_array)):
        for j in range(len(out_array[0])):
            if borderMean - tolerance <= pic_array[i][j] and borderMean + tolerance >= pic_array[i][j]:
                out_array[i][j] = 0
            elif pic_array[i][j] > borderMean:
                out_array[i][j] = 0 #blackens blood vessels
            elif pic_array[i][j] < borderMean and regions.getCompartment(i, j).noise_compartment == False:
                out_array[i][j] = WHITE
            #if regions.getCompartment(i, j).noise_compartment == True:
            #   out_array[i][j] = WHITE // 2



def findBoundaryPoints(binary):
    boundary = []
    for i in range(len(binary)):
        for j in range(len(binary[0])):
            n_touching = len(list(filter(lambda p: binary[p[0]][p[1]] == WHITE, getNeighborIndices(binary, i, j))))
            #number of neighbors that are within a potential cell
            if binary[i][j] == WHITE and n_touching != 8:
                boundary.append((i, j))#, n_touching > 5)) #last argument is cusp boolean
    return boundary

def internalBorderTest(pic_array, out_array, boundary):
    internalBorder = out_array[:] #pseudo-deep copy
    boundaryIndices = [(b[0], b[1]) for b in boundary] #no cusp boolean
    tolerance = 0.15 * WHITE #lenient
    for i in range(len(pic_array)):
        for j in range(len(pic_array[0])):
            if pic_array[i][j] > borderMean - tolerance and (i, j) not in boundaryIndices:
                internalBorder[i][j] = 0
    return internalBorder

class Cluster:

    clusters = []

    def __init__(self, binary, boundary):

        self.boundary = boundary #DO NOT SORT THIS! Order is important
        #self.boundary2D = self.getBoundary2D()
        self.binary = binary
        self.cells = []
        self.cusps = []
        self.pivot = (np.mean([p[0] for p in self.boundary]), np.mean([p[1] for p in self.boundary]))

        if not isinstance(self, Cell):
            Cluster.clusters.append(self)

    def getBoundary2D(self):
        sortedBound = self.boundary[:]
        sortedBound.sort() #default key sort tuples (i, j) by i then j
        return [list(filter(lambda p: p[0] == x, sortedBound) for x in range(sortedBound[0][0], sortedBound[-1][0] + 1))]


    def getTrueCusps(self, segmentLen=3):
        #print(self.boundary)
        assert len(self.boundary) > segmentLen * 3, 'boundary is too short. consider killing cluster'
        cusps = []

        for point in self.boundary: #filter(lambda p: p[2], self.boundary):
            k = self.boundary.index(point)
            before = self.boundary[k - segmentLen]
            try:
                after = self.boundary[k + segmentLen]
            except IndexError:
                after = self.boundary[k + segmentLen - len(self.boundary)]

            notCusp = False
            midpt = (math.floor(np.mean([p[0] for p in [point, before, after]])), math.floor(np.mean([p[1] for p in [point, before, after]])))
            if self.binary[midpt[0]][midpt[1]] != 0:
                notCusp = True

            if notCusp:
                continue

            #Recall i axis is REVERSED on coordinate map

            ldy = - (point[0] - before[0]); ldx = (point[1] - before[1])
            if ldx == 0:
                left_deriv = math.inf #1000 if ldy > 0 else -1000
            else:
                left_deriv = ldy / ldx

            rdy = - (after[0] - point[0]); rdx = (after[1] - point[1])
            if rdx == 0:
                right_deriv = math.inf #1000 if rdy > 0 else -1000
            else:
                right_deriv = rdy / rdx

            angle = abs( math.atan(left_deriv) - math.atan(right_deriv) )

            if angle < 0.75 * math.pi:
                cusps.append(point)

        self.cusps = cusps
        return cusps
        
    def pruneCusps(self, cusps):
        cusps = cusps[:] # pseudo-deep copy
        arcs = []
        while cusps:
            seq = []
            first = cusps.pop(0)
            k = 0
            while max(abs(cusps[k][0] - first[0]), abs(cusps[k][1] - first[k][1])) == 1:
                seq.append(cusps[k])
                k += 1
            arcs.append(seq)
            del cusps[:k]
        for arc in arcs:
            if len(arc) < 3:
                arcs.remove(arc)
        return arcs
            


    def showCusps(self, *args):
        cusps = self.getTrueCusps(*args)
        for c in cusps:
            for n in getNeighborIndices(self.binary, c[0], c[1]):
                self.binary[n[0]][n[1]] = WHITE // 2



    def propagateInternalBoundaries(self):
        #cuspPoints = list(filter(lambda p: p[2], self.boundary))
        cuspPoints = self.cusps
        if len(cuspPoints) <= 1:
            return None

        edges = []
        for cp in cuspPoints:
            edge = [cp]
            # and p not in self.boundary
            leader = max( filter(lambda p: self.binary[p[0]][p[1]] == WHITE, getNeighborIndices(self.binary, cp[0], cp[1])), key=lambda p: self.pic[p[0]][p[1]])
            previous = cp
            exclude = []
            while leader not in cuspPoints:#self.boundary:
                edge.append(leader)
                self.binary[leader[0]][leader[1]] = 0
                try:
                    neighbors = getForwardNeighbors(self.binary, previous, leader)
                except AssertionError:
                    print("Previous ", previous, " | Point ", leader)
                    break
                else:
                    previous = leader
                    options = list(filter(lambda p: self.binary[p[0]][p[1]] == WHITE and p not in edge and p not in exclude and p not in self.boundary, neighbors))
                    try:
                        leader = max(options, key=lambda p: self.pic[p[0]][p[1]])
                    except ValueError as e:
                        print((str(e)))
                        exclude = neighbors
                        break
                #finally:
                #    exclude = neighbors
            edges.extend(edge)

        #for ep in edges:
        #    self.binary[ep[0]][ep[1]] = 0




    def kill(self):
        Cluster.clusters.remove(self)

class Cell(Cluster):

    def __init__(self, binary, boundary):
        super.__init__(self, binary, boundary)
        self.points = self.getPoints()

    def getPoints(self):
        pass

    def area(self):
        pass

    def eccentricity(self):
        pass



def makeClusters(binary, boundary):
    
    if len(boundary) == 0:
        return
    boundary.sort() #should be sorted but double-checking; sort by i then j
    clusterBounds = []
    while boundary:
        clusterBounds.append([boundary[0]])
        pivot = boundary.pop(0)
        current = clusterBounds[-1] #last cluster element
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
                    k = -2
                    while abs(k) <= len(current):
                        if (current[k][0], current[k][1]) in neighbors:
                            del current[k+1:]
                            break
                        k -= 1
                    #the point was part of the boundary where edge thickness was > 1 pixel and is therefore not a useful neighbor
                    current.pop()
                else:
                    break #exits outer while loop
            else:
                current.append(neighbor)
                boundary.remove(neighbor)
        if len(current) < 12:
            clusterBounds.remove(current)

    clusters = []
    for c in clusterBounds:
        clusters.append(Cluster(binary, c))
    print("$$$$$$$$$$$$$$", len(clusterBounds))
    return clusters


def process_image(inFile):

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
        clusters = makeClusters(out_array, boundary)
        i = -1
        for c in clusters:
            i += 1
            try:
                #c.showCusps(7)
                c.getTrueCusps(7)
            except AssertionError:
                print(i, 'AssertionError')
                pass
            finally:
                c.propagateInternalBoundaries()
        skimage.external.tifffile.imsave(inFile.replace('.tif', '_BinaryEdged.tif'), out_array)


        # print(len(clusters))
        # for c in clusters:
        #     print(c)
        
        # for k in range(len(clusters)):
        #     ck = [(c[0], c[1]) for c in clusters[k].boundary]
        #     c_arr = out_array[:]
        #     for i in range(len(c_arr)):
        #         for j in range(len(c_arr[0])):
        #             if (i,j) in ck:
        #                 c_arr[i][j] = WHITE
        #             else:
        #                 c_arr[i][j] = 0
        #     skimage.external.tifffile.imsave(inFile.replace('.tif', '_c'+str(k)+'.tif'), c_arr)
        #print(Cluster.clusters, len(Cluster.clusters))


#class Stack:

def collateSlices():
    pass



#prefixes = ['/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p1f2_normal/eye1-',
#'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p1f3_normal/eye1-',
#'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p2f1_normal/eye1-',
#'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p2f2_normal/eye1-',
#'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p2f3_normal/eye1-',
#'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye2p1f1_normal/eye2-',
#'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye2p1f2_normal/eye2-',
#'/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye2p1f3_normal/eye2-']

prefix = '/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/Mouse 1/eye1p1f1_normal/eye1-'

def parallel(prefix):

    #make a stack object somewhere
    x = 24
    while True:
        try:
            process_image(prefix + str(x).rjust(4, '0') + '.tif')
        except IOError:
            break
        else:
            print(prefix)
            x += 1
            break

#with Pool(5) as p:
#   p.map(parallel, prefixes)
parallel(prefix)






