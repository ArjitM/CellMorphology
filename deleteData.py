import sys
import os

fol = sys.argv[1]
for d in next(os.walk(fol))[1]:
    for f in os.listdir(fol + '/' + d):
        if "_" in f or "largest" in f or "csv" in f or "overlaid" in f:
            #print(f)
            os.remove("{0}/{1}/{2}".format(fol, d, f))