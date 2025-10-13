import re
import os


def getAcqData():
    data = dict()
    pattern = re.compile(r"^(?P<pid>ISPY1_\d+)_.+_vis(?P<vis>\d+)_acq(?P<acq>\d+)")
    for fname in os.listdir("Datasets/BreastDCEDL_spy1/spt1_dce"):
        match = pattern.search(fname)
        if match:
            groupdict = match.groupdict()
            pid = groupdict["pid"]
            vis = groupdict["vis"] 
            acq = groupdict["acq"]
            
            if not os.path.exists(f"Datasets/BreastDCEDL_spy1/spy1_mask/{pid}_spy1_vis1_mask.nii.gz"):
                continue
            
            if vis!="1":
                raise
            if pid not in data:
                data[pid]=set()
            data[pid].add(acq)
    count4 = 0
    count6 = 0
    countLess = 0
    for pid,acqs in data.items():
        match len(acqs):
            case 3:
                continue
            case 4:
                count4+=1
                data[pid] = ["0","1","2"]
            case 6:
                count6+=1
                data[pid] = ["0","2","5"]
            case _:
                countLess +=1
                data[pid] = None
    data = {pid:acq for pid,acq in data.items() if acq is not None}
    print(f"{count4=} {count6=} {countLess=}")
    return data

getAcqData()