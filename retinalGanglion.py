from multiprocessing import Pool
import numpy as np 
#from __future__ import division

import skimage
from skimage import io
from skimage import external
from skimage import morphology
from skimage import filters
from skimage import color
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
        self.border_mean = 0

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
            self.border_mean = np.mean(border)
            return np.mean(border)

        self.hasBorder = False
        #print(self.str())
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
            if hits > (len(c.neighbors) // 2):
                c.set_noise_compartment(True)
            else:
                c.set_noise_compartment(False)

    def getAvgBorder(self, binary):
        not_Noise = filter(lambda c: not bool(c.noise_compartment), self.compartments)      
        avgs = [c.avgBorder(binary) for c in not_Noise if c.hasBorder]
        return np.nanmean([a for a in avgs if a])


    def getAvgBorders(self, binary):
        not_Noise = filter(lambda c: not bool(c.noise_compartment), self.compartments)      
        return [c.avgBorder(binary) for c in not_Noise if c.hasBorder]

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
            #cut = np.std(out_array) // 3

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
    tolerance = 0.085 * WHITE # +/- 7 % of total image range

    for i in range(len(out_array)):
        for j in range(len(out_array[0])):
            local_border_avg = regions.getCompartment(i, j).border_mean
            if local_border_avg - tolerance <= pic_array[i][j] and local_border_avg + tolerance >= pic_array[i][j]:
                out_array[i][j] = 0
            elif pic_array[i][j] > local_border_avg:
                out_array[i][j] = 0 #blackens blood vessels
            elif pic_array[i][j] < local_border_avg and regions.getCompartment(i, j).noise_compartment == False:
                out_array[i][j] = WHITE
            #if regions.getCompartment(i, j).noise_compartment == True:
               #out_array[i][j] = WHITE // 2



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

class Cusp:

    def __init__(self, point, left_deriv, right_deriv, angle, angle_trend):
        self.point = point
        self.left_deriv = left_deriv
        self.right_deriv = right_deriv
        self.angle = abs(angle)
        self.angle_trend = angle_trend

    #def angle(self):
    #    return abs( math.atan(self.left_deriv) - math.atan(self.right_deriv) ) #in radians!!

class Edge:

    def __init__(self, edge):
        self.start = edge.pop(0)
        self.end = edge.pop()
        self.internalEdge = edge


class Cluster:

    clusters = []

    def __init__(self, binary, boundary, stack_slice, internalEdges=[]):

        self.boundary = boundary #DO NOT SORT THIS! Order is important
        #self.boundary2D = self.getBoundary2D()
        self.binary = binary
        self.cells = []
        self.cusps = []
        self.pivot = (np.mean([p[0] for p in self.boundary]), np.mean([p[1] for p in self.boundary]))
        self.constriction_points = []
        self.internalEdges = internalEdges
        self.stack_slice = stack_slice

        if not isinstance(self, Cell):
            Cluster.clusters.append(self)

    def getTrueCusps(self, segmentLen=8):
        #print(self.boundary)
        assert len(self.boundary) > segmentLen * 3, 'boundary is too short. consider killing cluster'
        cusps = []

        for point in self.boundary: #filter(lambda p: p[2], self.boundary):
            k = self.boundary.index(point)

            angles = []
            notCusp = False

            for segmentPoint in range(segmentLen // 2, 1 + segmentLen):
                
                before = self.boundary[k - segmentPoint]
                try:
                    after = self.boundary[k + segmentPoint]
                except IndexError:
                    after = self.boundary[k + segmentPoint - len(self.boundary)]

                midpt = (math.floor(np.mean([p[0] for p in [before, after]])), math.floor(np.mean([p[1] for p in [before, after]])))
                if self.binary[midpt[0]][midpt[1]] != 0:
                    #notCusp = True
                    continue 

                ldy = - (point[0] - before[0]); ldx = (point[1] - before[1])
                if ldx == 0:
                    left_deriv = math.inf if ldy > 0 else (- math.inf) #1000 if ldy > 0 else -1000
                    #continue
                else:
                    left_deriv = ldy / ldx

                rdy = - (after[0] - point[0]); rdx = (after[1] - point[1])
                if rdx == 0:
                    right_deriv = math.inf if rdy > 0 else (- math.inf) #1000 if rdy > 0 else -1000
                    #continue
                else:
                    right_deriv = rdy / rdx

                angles.append(abs( math.atan(left_deriv) - math.atan(right_deriv) ))

            if notCusp:
                continue

            if angles == []:
                continue
            angle = np.nanmean(angles)

            if angle < 0.75 * math.pi:
                cusps.append(Cusp(point, left_deriv, right_deriv, angle, angles))

        self.cusps = cusps
        return cusps
        
    def pruneCusps(self):
        # cusps = [c.point for c in self.cusps]
        arcs = []
        while self.cusps:
            seq = []
            previous = self.cusps[0]
            k = 0
            #arc is a sequence of contiguous (max distance 1 pixel) cusp-points
            while k < len(self.cusps) and max(abs(self.cusps[k].point[0] - previous.point[0]), abs(self.cusps[k].point[1] - previous.point[1])) <= 1:
                seq.append(self.cusps[k])
                previous = self.cusps[k]
                k += 1
            arcs.append(seq)
            del self.cusps[:k]

        self.arcs = [arc for arc in arcs if len(arc) >= 1] #removing arcs with len < 3 DOES NOT work!
        return self.arcs

    def showCusps(self, *args):
        #for c in self.constriction_points:
        for arc in self.arcs:
            for c in arc:
                self.binary[c.point[0]][c.point[1]] = WHITE // 2
        for c in self.constriction_points:
            for n in getNeighborIndices(self.binary, c.point[0], c.point[1]):
                self.binary[n[0]][n[1]] = 0


    def splitBentCells(self):
        pass

    @staticmethod
    def makeCell(stack_slice, binary, cell_boundary, internalEdges):
        cell = Cell(stack_slice, binary, cell_boundary, internalEdges)
        return cell

        

    def propagateInternalBoundaries(self):
        #cuspPoints = list(filter(lambda p: p[2], self.boundary))
        #cuspPoints = self.cusps
        
        if len(self.arcs) <= 1:
            return None

        self.constriction_points = []
        
        cleave_points = [ min(arc, key=lambda c: c.angle) for arc in self.arcs]
        # these are the points where internal boundaries start/stop. Find by looking for cusp region points with the least (most constricted) angle

        completed_pairs = []
        #boundaries that have already been made, avoid duplication

        for cp in cleave_points:

            self.constriction_points.append(cp)
            orientation = lambda p: np.mean( [math.pi - abs(math.atan(p.left_deriv) - math.atan(cp.left_deriv)), math.pi - abs(math.atan(p.right_deriv) - math.atan(cp.right_deriv))] )
            viable = filter(lambda p: orientation(p) < math.pi * 0.75, cleave_points)
            viable_no_duplicate = filter(lambda p: (p, cp) not in completed_pairs and (cp, p) not in completed_pairs, viable)
            try:
                pair = min(viable_no_duplicate, key = lambda p: (cp.point[0] - p.point[0])**2 + (cp.point[1] - p.point[1])**2 )
            except ValueError:
                continue
            #print("Pair", pair.point)
            completed_pairs.append((pair, cp))

            delta_i = cp.point[0] - pair.point[0]
            ki = -1 if delta_i > 0 else 1
            delta_i = abs(delta_i)

            delta_j = cp.point[1] - pair.point[1]
            kj = -1 if delta_j > 0 else 1
            delta_j = abs(delta_j)

            # to propagate boundaries from one constriction site to another, assume a simple diagonal line (m=1)
            ## and use periodic shifts to complete the boundary
            edge = []
            edge.append(cp.point)
            #print("Start: ", cp.point)

            if delta_i > delta_j:

                #print("delta i > j")

                #vert_shifts = delta_i - delta_j
                if delta_j != 0:
                    shift_period = delta_i // delta_j #vert_shifts #need a shift every length period of i 
                    shift = True

                else:
                    shift_period = 100
                    shift = False


                #print("horizontal shift period", shift_period)

                x, y = cp.point[1], cp.point[0]
                k = 0
                x_rem, y_rem = math.inf, math.inf
                while min(x_rem, y_rem) > 1:

                    #print("k", k)
                    k += 1
                    y += ki
                    
                    if shift and k % shift_period == 0:
                        x += kj #horizontal shift
                    edge.append((y, x))

                    if k > 500:
                        break
                    #print((y,x))

                    x_rem = abs(x - pair.point[1])
                    y_rem = abs(y - pair.point[0])

                assert x_rem <= 1 or y_rem <= 1, 'your code is trash'

                if x_rem > 1:
                    for rem in range(x, pair.point[1], kj):
                        edge.append((y, rem))
                        #print((y, rem))

                else:
                    for rem in range(y, pair.point[0], ki):
                        edge.append((rem, x))
                        #print((rem, x))

                

            elif delta_j > delta_i:

                #print('delta j > i')
                #horiz_shifts = delta_j - delta_i

                if delta_i != 0:
                    shift = True
                    shift_period = delta_j // delta_i #need a shift every length period of j 

                else:
                    shift_period = 100
                    shift = False

                #print("vertical shift period", shift_period)

                x, y = cp.point[1], cp.point[0]
                k = 0
                x_rem, y_rem = math.inf, math.inf
                while min(x_rem, y_rem) > 1:

                    #print('k', k)
                    k += 1
                    x += kj

                    if shift and k % shift_period == 0:
                        y += ki #vertical shift
                    edge.append((y, x))
                    
                    if k > 500:
                        break
                    #print((y,x))

                    x_rem = abs(x - pair.point[1])
                    y_rem = abs(y - pair.point[0])

                assert x_rem <= 1 or y_rem <= 1, 'your code is trash'

                if x_rem > 1:
                    for rem in range(x, pair.point[1], kj):
                        edge.append((y, rem))
                        #print((y, rem))

                else:
                    for rem in range(y, pair.point[0], ki):
                        edge.append((rem, x))
                        #print((rem, x))
            
            else:

                #print("i = j")
                x, y = cp.point[1], cp.point[0]
                k = 0
                while min(abs(x - pair.point[1]), abs(y - pair.point[0])) > 1:
                    #print('k', k)
                    k += 1
                    y += ki
                    x += kj
                    edge.append((y, x))

                    if k>500:
                        break
                    #print((y,x))
            edge.append(pair.point)

            for p in edge:
                self.binary[p[0]][p[1]] = 0

            #print("New edge done")

            self.internalEdges.append(Edge(edge))

        #for ep in edges:
        #    self.binary[ep[0]][ep[1]] = 0

    def splitByEdges(self):
        if self.internalEdges == []:
            newCell = Cluster.makeCell(self.stack_slice, self.binary, self.boundary, [])
            if len(newCell.boundary) > 10:
                self.stack_slice.addCell(newCell)
        else:
            if self.internalEdges:
                divider = self.internalEdges.pop()
                try:
                    start_index = self.boundary.index(divider.start)
                except ValueError: #intersecting internal edges
                    return
                
                try:
                    end_index = self.boundary.index(divider.end)
                except ValueError:
                    return

                if start_index > end_index:
                    cell_1_bound = list(reversed(divider.internalEdge)) + self.boundary[start_index : end_index : -1] + [divider.end]
                    cell_2_bound = list(reversed(divider.internalEdge)) + self.boundary[start_index:] + self.boundary[:end_index+1]

                else:
                    cell_1_bound = list(reversed(divider.internalEdge)) + self.boundary[start_index : end_index : 1] + [divider.end]
                    cell_2_bound = list(reversed(divider.internalEdge)) + self.boundary[end_index::1] + self.boundary[:start_index+1:1]

                cell_1 = Cluster.makeCell(self.stack_slice, self.binary, cell_1_bound, self.internalEdges)
                cell_2 = Cluster.makeCell(self.stack_slice, self.binary, cell_2_bound, self.internalEdges)

                cell_1.splitByEdges()
                cell_2.splitByEdges()

                #if isinstance(self, Cell):
                #    Cell.kill(self) 


    def kill(self):
        Cluster.clusters.remove(self)

class Cell(Cluster):

    def __init__(self, stack_slice, binary, boundary, internalEdges=[]):
        super().__init__(binary, boundary, stack_slice, internalEdges)
        #stack_slice.addCell(self)
        if not self.internalEdges:
            self.area = self.area()
            self.roundness = self.roundness()
            self.colored = skimage.color.grey2rgb(self.binary)


    def pointWithin(self, point):
        y = point[0]
        x = point[1]
        y_bounds = [p for p in self.boundary if p[1] == x]
        x_bounds = [p for p in self.boundary if p[0] == y]

        if y_bounds == [] or x_bounds == []:
            return False
        if point in y_bounds or point in x_bounds:
            return True
        y_bounds.sort(key = lambda b: b[0])
        x_bounds.sort(key = lambda b: b[1])
        
        if point[0] < y_bounds[0][0] or point[0] > y_bounds[-1][0] or point[1] < x_bounds[0][1] or point[1] > x_bounds[-1][1]:
            return False

        y_pairs = self.get_var_pairs(y_bounds, 0)
        x_pairs = self.get_var_pairs(x_bounds, 1)

        def within_var_bounds(var_pairs, point, index):
            #index 0 for y, 1 for x
            for var_pair in var_pairs:
                if point[index] >= var_pair[0][index] and point[index] <= var_pair[1][index]:
                    return True
            return False

        return within_var_bounds(y_pairs, point, 0) and within_var_bounds(x_pairs, point, 1)


    def getBoundary2D(self):
        sortedBound = self.boundary[:]
        sortedBound.sort() #default key sort tuples (i, j) by i then j
        return [ list(filter(lambda p: p[0] == y, sortedBound)) for y in range(sortedBound[0][0], sortedBound[-1][0] + 1)]

    def get_var_pairs(self, var_bounds, index):
        if len(var_bounds) <= 1:
            return []
        k = 0
        var_pairs = []
        while k < len(var_bounds) - 1:
            interrupted = False
            for btwn in range(var_bounds[k][index], var_bounds[k+1][index]):
                #toggle = int(not(bool(index))) #toggle btwn 0 and 1
                if index == 1:
                    if self.binary[var_bounds[k][0]][btwn] == 0:
                        interrupted = True
                        break
                else:
                    if self.binary[btwn][var_bounds[k][1]] == 0:
                        interrupted = True
                        break
            if not interrupted:
                var_pairs.append([var_bounds[k], var_bounds[k+1]])
            k += 1
        return var_pairs

    def contains_or_overlaps(self, other_cell):
        k = len(other_cell.boundary) // 8
        hits = 0
        for i in range(8):
            if self.pointWithin(other_cell.boundary[k * i]):
                hits += 1        
        return hits >= 4, hits >= 1


    def area(self):
        bounds = self.getBoundary2D()
        area = 0
        for b in bounds:
            x_pairs = self.get_var_pairs(b, 1)
            if x_pairs == []:
                area += 1
            else:
                for xp in x_pairs:
                    area += (xp[1][1] - xp[0][1] + 1)
        return area


    def roundness(self):
        circum = len(self.boundary)
        ideal = circum / math.pi
        k = circum // 8
        opposing = [[0,4], [1,5], [2,6], [3,7]]
        diameters = []

        def distance(p1, p2):
            d = math.sqrt( (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            return math.floor(d)

        for op in opposing:
            diameters.append(distance(self.boundary[op[0] * k], self.boundary[op[1] * k]))

        avgDiameter = np.mean(diameters)
        self.roundness = 1 - (abs(avgDiameter - ideal) / ideal)
        return self.roundness


    def kill(self):
        self.stack_slice.removeCell(self)



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


        # print(len(clusters))
        # for c in clusters:
        #     print(c)
        
        # c_arr = out_array[:]
        # for i in range(len(c_arr)):
        #     for j in range(len(c_arr[0])):
        #         c_arr[i][j] = 0
                
        # for k in range(len(clusters)):
        #     ck = [(c[0], c[1]) for c in clusters[k].boundary]

            
        #     for i in range(len(c_arr)):
        #         for j in range(len(c_arr[0])):
        #             if (i,j) in ck:
        #                 c_arr[i][j] = WHITE
        #             else:
        #                 c_arr[i][j] = 0
        #     skimage.external.tifffile.imsave(inFile.replace('.tif', '_cluster_' +str(k)+'.tif'), c_arr)
        # #print(Cluster.clusters, len(Cluster.clusters))


class Stack_slice:

    def __init__(self, number, cells):
        self.cells = cells
        self.number = number
        self.finalizedCellSlice = Stack_slice_largest(number, [])

    def addCell(self, cell):
        if isinstance(cell, Cell):
            self.cells.append(cell)
        else:
            print("not a cell instance")

    def removeCell(self, cell):
        self.cells.remove(cell)

    def pruneCells(self, roundness_thresh=0.75):
        self.cells = [c for c in self.cells if c.roundness > roundness_thresh]
        self.cells = [c for c in self.cells if c.area > 25]

class Stack_slice_largest(Stack_slice):

    def __init__(self, number, cells):
        self.cells = cells
        self.number = number


class Stack:

    def __init__(self, stack_slices=[]):
        self.stack_slices = stack_slices
        self.large_Cells = []

    def addSlice(self, stack_slice):
        self.stack_slices.append(stack_slice)

    def collate_slices(self):
        for stack_slice in self.stack_slices:

            for cell in stack_slice.cells:
                hits = 0
                large_replace = []

                for large_Cell in self.large_Cells:
                    if cell.contains_or_overlaps(large_Cell)[0]:
                        large_replace = large_Cell
                        hits += 1
                    if hits > 1:
                        break

                if hits > 1: #limit reached
                    large_replace = None
                    continue

                elif hits == 1:
                    self.large_Cells.remove(large_replace)
                    self.large_Cells.append(cell)

                else:
                    new_cell = True
                    for large_Cell in self.large_Cells:
                        (contained, overlapping) = large_Cell.contains_or_overlaps(cell)
                        if contained: #contained
                            new_cell = False
                            break
                        if overlapping: #overlapping
                            if large_Cell.area > cell.area:
                                new_cell = False
                                break
                            else:
                                large_replace.append(large_Cell)

                    if new_cell:
                        self.large_Cells.append(cell)
                    if large_replace:
                        #print('replaced')
                        for lr in large_replace:
                            self.large_Cells.remove(lr)

        for lg in self.large_Cells:
            lg.stack_slice.finalizedCellSlice.addCell(lg)



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




