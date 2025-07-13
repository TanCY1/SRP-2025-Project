import pandas as pd

def printClosest(ar1, ar2, m, n):
    i = 0
    j = 0
    min_diff = float('inf')
    closest_pair = (None, None)

    while i < m and j < n:
        diff = abs(ar1[i] - ar2[j])

        if diff < min_diff:
            min_diff = diff
            closest_pair = (ar1[i], ar2[j])

        # Move pointers to try and reduce diff
        if ar1[i] < ar2[j]:
            i += 1
        else:
            j += 1
    print(f"Closest pair: {closest_pair} with difference {min_diff}")

metadata = pd.read_csv("BreastDCEDL_spy1_metadata.csv")

count0 = metadata.pCR.value_counts()[0]
count1 = metadata.pCR.value_counts()[1]

print(metadata.pCR.value_counts())


#1, 3
#4, 11  (12)
