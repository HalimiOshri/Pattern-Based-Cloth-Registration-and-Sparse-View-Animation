import os
import torch
import imageInterpolater

sequence = 's--20210823--1323--0000000--pilot--patternCloth'

step = 1
first = 488
last = 4450

meshFormat = '/mnt/home/oshrihalimi/cloth_completion/data_processing/{}/codecResClothesUnposed/{{:06d}}.ply'.format(sequence)
outputFile = '/mnt/home/oshrihalimi/cloth_completion/data_processing/{}/clothesCodec/clothes_unposed_mean.ply'.format(sequence)

frameRange = range(first, last + 1, step)
numFrames = len(frameRange)

s = 0.0

for frameId in frameRange:
    print(frameId)
    meshFile = meshFormat.format(frameId)
    v, f = imageInterpolater.loadplyvet(meshFile)
    s = s + (v / numFrames)

imageInterpolater.saveplymesh(outputFile, s, f)
