import math
import numpy as np
import skimage
from skimage import io
from skimage import external
from skimage import morphology
from skimage import filters
from skimage import color
from Binarize import *
import scipy.ndimage

global WHITE
WHITE = None

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

    def transformToCell(self):
        nc = Cluster.makeCell(self.stack_slice, self.binary, self.boundary, [])
        self.stack_slice.addCell(nc)
        return nc

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
        
        #cleave_points = [ min(arc, key=lambda c: c.angle) for arc in self.arcs]
        # these are the points where internal boundaries start/stop. Find by looking for cusp region points with the least (most constricted) angle
        cleave_points = [arc[len(arc) // 2] for arc in self.arcs]


        completed_pairs = []
        #boundaries that have already been made, avoid duplication

        for cp in cleave_points:

            self.constriction_points.append(cp)
            orientation = lambda p: np.mean( [math.pi - abs(math.atan(p.left_deriv) - math.atan(cp.left_deriv)), math.pi - abs(math.atan(p.right_deriv) - math.atan(cp.right_deriv))] )
            viable = filter(lambda p: orientation(p) < math.pi * 0.5, cleave_points)
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
                self.binary[p[0]][p[1]] = 1 #making this 0 intereferes with area

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
        #self.roundness = 0
        if not self.internalEdges:
            self.internalBoundaryHits = 0 #if cell is in "controversial" lots of boundaries region, it is killed
            _ = self.area #invoke property to update interior
            self.interior = [] #this is updated by the area function
            #self.center = scipy.ndimage.measurements.center_of_mass(self.interior + self.boundary)
            #self.area = self.area()
            if self.internalBoundaryHits > self.area / 3:
                self.kill()
            #else:
            #    self.roundness = self.getRoundness()
            # self.colored = skimage.color.grey2rgb(self.binary)


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


    def addInternalBoundaryHit(self):
        self.internalBoundaryHits += 1

    def get_var_pairs(self, var_bounds, index, area=False): #if area is true check for controversial region
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
                    if area and self.binary[var_bounds[k][0]][btwn] == 1:
                        self.addInternalBoundaryHit()
                else:
                    if self.binary[btwn][var_bounds[k][1]] == 0:
                        interrupted = True
                        break
            if not interrupted:
                var_pairs.append([var_bounds[k], var_bounds[k+1]])
            k += 1
        return var_pairs

    def contains_or_overlaps(self, other_cell):
        k = len(other_cell.boundary)
        hits = 0
        for other_p in other_cell.boundary:
            if self.pointWithin(other_p):
                hits += 1        
        return hits >= k // 3 and self.pointWithin(other_cell.pivot), hits >= k // 6

    @property
    def area(self):
        self.interior = []
        bounds = self.getBoundary2D() #returns row-wise 2d matrix
        area = 0
        for b in bounds: #boundary points in each row
            x_pairs = self.get_var_pairs(b, 1, True)
            if x_pairs == []:
                area += 1
            else:
                for xp in x_pairs:
                    area += (xp[1][1] - xp[0][1] + 1)
                    self.interior.extend([(xp[0][0], k) for k in range(xp[0][1], xp[1][1]+1)])
        return area


    @property
    def roundness(self):
        circum = len(self.boundary)
        ideal = circum / (2 * math.pi)
        # k = circum // 8
        # opposing = [[0,4], [1,5], [2,6], [3,7]]
        # diameters = []

        def distance(p1, p2):
            d = math.sqrt( (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            return d

        normalized_distances = list(np.sqrt([ ((distance(b, self.pivot) - ideal)/ideal)**2 for b in self.boundary]))

        return 1 - np.mean(normalized_distances)
        #return self.roundness


    def kill(self):
        if self in self.stack_slice.cells:
            self.stack_slice.removeCell(self)
