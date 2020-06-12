#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 21:52:10 2019

@author: legendary_yin
"""

# Graph
infi = 10000
G = [[0, 6, infi, 1, infi],
     [6, 0, 5, 2, 2],
     [infi, 5, 0, infi, 5],
     [1, 2, infi, 0, 1],
     [infi, 2, 5, 1, 0]]


result = [[0, infi, 0],
          [1, infi, 1],
          [2, infi, 2],
          [3, infi, 3],
          [4, infi, 4]]

origin = 0
visited = [origin]

result[origin][1] = 0

while len(set(visited)) != len(G):
    
    for i in range(len(G[visited[-1]])):
        if result[i][1] > G[visited[-1]][i] + result[visited[-1]][1]:
            result[i][1] = G[visited[-1]][i] + result[visited[-1]][1]
            result[i][2] = visited[-1]
            
    # find the current minimum distance    
    minimum = infi
    minindex = origin
    for j in range(len(G)):
        if j not in visited and result[j][1] < minimum:
            minindex = j
        
    visited.append(minindex)

for i in range(len(result)):
    if i != origin:
        stack = []
        thisnode = i
        while thisnode != origin:
            stack.append(thisnode)
            thisnode = result[thisnode][2]
        
        print([origin] + stack[::-1])
        
# floyd
G1 = G
via = [[-1] * len(G1),
       [-1] * len(G1),
       [-1] * len(G1),
       [-1] * len(G1),
       [-1] * len(G1)]

for i in range(len(G1)):
    for j in range(len(G1)):
        for k in range(len(G1)):
            if G1[j][k] > G1[j][i] + G1[i][k]:
                G1[j][k] = G1[j][i] + G1[i][k]
                via[j][k] = i
                
G1
via

origin = 0
destination = 2

def printresult(a,b,via1):
    if via1[a][b] == -1:
        print(a)
        #print(b)
    else:
        printresult(a,via1[a][b],via1)
        printresult(via1[a][b],b,via1)

printresult(origin,destination,via)
        
        
        
    
            
            