import os
import multiprocessing
from multiprocessing import Pool

locations = [
'../vit A/vit_A_free/',
'../WT/',
'../RD1-P2X7KO/',
'../RD1/',
'../VAF_new_cohort/'
]

prefixes = []
for loc in locations:
    print(loc)
    for dir1 in next(os.walk(loc))[1]: #Expt #/Mouse#
        try:
            for dir2 in next(os.walk(loc + dir1))[1]:
                if 'normal' in dir2:
                    prefixes.append(loc + dir1 + "/" + dir2 + "/")
                else:
                    for dir3 in next(os.walk(loc + dir1 + "/" + dir2))[1]:
                        if 'normal' in dir3:
                            prefixes.append(loc + dir1 + "/" + dir2 + "/" + dir3 + "/")
        except:
            pass

def onearg(d):
    print(d)
    for f in next(os.walk(d))[2]:
        if "_BinaryPivots" in f or "_segmented.pkl" in f:
            os.system("rclone copy '{0}{1}' 'arjit_bdrive:/CellMorph2-12/{2}' --ignore-existing".format(d, f, d.replace('../','')))

#print(prefixes)
with Pool(50) as p:
    p.map(onearg, prefixes)
