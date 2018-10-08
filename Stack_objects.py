from Cell_objects import *


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
        self.cells = [c for c in self.cells if c.area > 15 and c.area < 500]

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
