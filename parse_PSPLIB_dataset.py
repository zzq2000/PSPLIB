import re
import pandas as pd

def parse_RCPSP(psplib_file):
    data = open(psplib_file)
    input = data.readlines()

    # Initialize the sets and parameters
    V = []  # Set of vertices
    A = []  # Set of arcs
    times = []  # Set of times

    K = []  # Set of commodities (modes in this context)
    r = []  # Resource amounts
    R = []

    graph = False
    time = False
    resource = False
    for i in input:
        if ("*********************************************" in i):
            graph = False
            time = False
            resource = False
        if (graph == True and not "--------------" in i):
            row = pd.Series(i).str.split().values[0]
            V.append(row[0])
            if row[1] not in K:
                K.append(row[1])
            for j in range(int(row[2])):
                A.append((row[0], row[3 + j]))
            
        if (time == True and not "--------------" in i):
            row = pd.Series(i).str.split().values[0]
            times.append(row[2])
            r.append(row[3:])
        
        if (resource == True and not "--------------" in i):
            R = pd.Series(i).str.split().values[0]

        if (i == "jobnr.    #modes  #successors   successors\n"):
            graph = True
        if (i == "jobnr. mode duration  R 1  R 2  R 3  R 4\n"):
            time = True
        if (i == "RESOURCEAVAILABILITIES:\n"):
            resource = True

    data.close()

    # make the times integers
    times = list(map(int, times))

    # add resources
    R = [R, ] + r[1:-1] + [R, ]


    return (V, A, times, K, R)