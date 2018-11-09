import os

locations = [
'/Volumes/Arjit_Drive/Output-10-22/vit_A_free/',
'/Volumes/Arjit_Drive/Output-10-22/WT/',
'/Volumes/Arjit_Drive/Output-10-22/RD1-P2X7KO/',
'/Volumes/Arjit_Drive/Output-10-22/RD1/',
'/Volumes/Arjit_Drive/Output-10-22/VAF_new_cohort/'
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

for p in prefixes:
	print(p)