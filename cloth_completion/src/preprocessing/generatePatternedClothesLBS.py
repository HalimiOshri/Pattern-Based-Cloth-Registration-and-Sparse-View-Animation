# create LBS model file of for patterned clothes which has been registered using Oshri's pipeline
import numpy as np
import igl
import trimesh
import json
from src.pyutils import io

sequence = 's--20210823--1323--0000000--pilot--patternCloth'

# sourceLBSFile = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/kinematic_tracking/small_pattern/averageman_artist_v01_beta/model/Codec-Body.json'
sourceLBSFile = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/kinematic_tracking/small_pattern/averageman_artist_v01_beta/model/FreeForm-Body.json'

alignedClothesFile = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/surface_deformable/incremental_registration/Meshes/skinned-004396.ply'
# alignedBodyFile = '/mnt/home/donglaix/s--20210823--1323--0000000--pilot--patternCloth/singleLayerCodecRes/000488.ply'  # use the codec res body

alignedBodyFile = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/kinematic_tracking/small_pattern/averageman_artist_v01_beta/momentum/free_form_high_res/Meshes/skinned-004396.ply'

templateClothesFile = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/model/seamlessTextured.obj'
outName = sequence
outputFile = '/mnt/home/donglaix/{}/clothesCodec/clothes_lbs/codec.json'.format(outName)

with open(sourceLBSFile) as f:
    sourceLBSData = json.load(f)

v, vt, faces, ft = io.load_obj(templateClothesFile)
vt = np.array(vt)
faces = np.array(faces)
ft = np.array(ft)

sourceLBSData['SkinnedModel']['Faces']['Indices'] = faces.reshape(-1).tolist()
sourceLBSData['SkinnedModel']['Faces']['TextureIndices'] = ft.reshape(-1).tolist()
sourceLBSData['SkinnedModel']['Faces']['Offsets'] = (np.arange(faces.shape[0] + 1) * 3).tolist()
sourceLBSData['SkinnedModel']['TextureCoordinates'] = vt.tolist()

alignedClothesMesh = trimesh.load(alignedClothesFile, process=False)
alignedBodyMesh = trimesh.load(alignedBodyFile, process=False)
bv = np.array(alignedBodyMesh.vertices)
bf = np.array(alignedBodyMesh.faces)
cv = np.array(alignedClothesMesh.vertices)

sqrD, I, C = igl.point_mesh_squared_distance(cv, bv, bf)
bcc = igl.barycentric_coordinates_tri(C, bv[bf[I, 0]], bv[bf[I, 1]], bv[bf[I, 2]])

"""  # load weights provided by Fabian for sanity check
I = []
bcc = []
with open('/mnt/home/donglaix/s--20210823--1323--0000000--pilot--patternCloth/clothesCodec/clothes_lbs/shirtToBodyBarincentricMap.txt') as f: 
    for line in f:
        i, p1, p2 = line.split()
        I.append(int(i))
        p1 = float(p1)
        p2 = float(p2)
        bcc.append([1.0 - p1 - p2, p1, p2])
I = np.array(I)
bcc = np.array(bcc)
"""

# copy the position, normal info from nearest body vertices
inRestPositions = np.array(sourceLBSData['SkinnedModel']['RestPositions'])
outRestPositions = inRestPositions[bf[I, 0]] * bcc[:, 0:1] + inRestPositions[bf[I, 1]] * bcc[:, 1:2] + inRestPositions[bf[I, 2]] * bcc[:, 2:3]
sourceLBSData['SkinnedModel']['RestPositions'] = outRestPositions.tolist()
inRestVertexNormals = np.array(sourceLBSData['SkinnedModel']['RestVertexNormals'])
outRestVertexNormals = inRestVertexNormals[bf[I, 0]] * bcc[:, 0:1] + inRestVertexNormals[bf[I, 1]] * bcc[:, 1:2] + inRestVertexNormals[bf[I, 2]] * bcc[:, 2:3]
sourceLBSData['SkinnedModel']['RestVertexNormals'] = outRestVertexNormals.tolist()

SkinningOffsets = [0]
SkinningWeights = []
# assert cv.shape[0] == c2bIndex.shape[0]
maxNumItem = 0
for ci in range(cv.shape[0]):
    bis = bf[I[ci]]
    boneDict = {}  # map from joint indices to weights
    for bci, bi in enumerate(bis):
        for itemIdx in range(sourceLBSData['SkinnedModel']['SkinningOffsets'][bi], sourceLBSData['SkinnedModel']['SkinningOffsets'][bi + 1]):
            boneIdx = sourceLBSData['SkinnedModel']['SkinningWeights'][itemIdx][0]
            boneWeight = sourceLBSData['SkinnedModel']['SkinningWeights'][itemIdx][1]
            if boneIdx in boneDict:
                boneDict[boneIdx] += boneWeight * bcc[ci, bci]
            else:
                boneDict[boneIdx] = boneWeight * bcc[ci, bci]
    for boneIdx, boneWeight in boneDict.items():
        SkinningWeights.append([boneIdx, boneWeight])
    numItems = len(boneDict)
    SkinningOffsets.append(SkinningOffsets[-1] + numItems)
    if numItems > maxNumItem:
        maxNumItem = numItems
sourceLBSData['SkinnedModel']['SkinningWeights'] = SkinningWeights
sourceLBSData['SkinnedModel']['SkinningOffsets'] = SkinningOffsets
print('maximal bone numbers: {}'.format(maxNumItem))

with open(outputFile, 'w') as f:
    json.dump(sourceLBSData, f)
