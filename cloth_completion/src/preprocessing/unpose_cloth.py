# unpose underlying body vertices to estiamte poses
import os

import torch
import numpy as np
import imageInterpolater
from src.pyutils import tensorIO
from src.kinematic_modeling.lbs.linear_blend_skinning import LinearBlendSkinning

sequence = 's--20210823--1323--0000000--pilot--patternCloth'
outputFormat = '/mnt/home/oshrihalimi/cloth_completion/{}/preprocessing/codecResClothesUnposed/{:06d}.ply'
motionFormat = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/kinematic_tracking/small_pattern/averageman_artist_v01_beta/momentum/skeleton_tracking/CompactPose/pose-{:06d}.txt'
vertexFormat = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/shirt_tracking/small_pattern/surface_deformable/incremental_registration/Meshes/skinned-{:06d}.ply'
step = 1
first = 0
last = 4450

lbs = {}
lbs["model_json_path"] = '/mnt/home/donglaix/s--20210823--1323--0000000--pilot--patternCloth/clothesCodec/clothes_lbs/codec.json'
lbs["scale_path"] = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/kinematic_tracking/small_pattern/averageman_artist_v01_beta/kinematic_alignment/body_personalization/personalized/final/scale.txt'
lbs["model_txt_path"] = '/mnt/home/fabianprada/Data/SociopticonProcessing/s--20210823--1323--0000000--pilot--patternCloth/kinematic_tracking/small_pattern/averageman_artist_v01_beta/model/averageman_v01_Body.cfg'
lbs["num_max_skin_joints"] = 9

os.makedirs(os.path.dirname('/mnt/home/oshrihalimi/cloth_completion/{}/preprocessing/codecResClothesUnposed/'.format(sequence)), exist_ok=True)  # output directory

if __name__ == '__main__':
    lbs_fn = LinearBlendSkinning(model_json_path=lbs["model_json_path"], model_txt_path=lbs["model_txt_path"], num_max_skin_joints=lbs["num_max_skin_joints"])

    with open(lbs["scale_path"]) as f:
        scale = torch.Tensor([float(x) for x in f.read().splitlines()]).expand(1, -1)
    global_scaling = torch.Tensor([[10., -10., -10.]])
    pose_ext = os.path.splitext(motionFormat)[1]

    for frameId in range(first, last + 1, step):
        print(frameId)
        vertexFile = vertexFormat.format(frameId)
        if not os.path.isfile(vertexFile):
            continue
        v, f = imageInterpolater.loadplyvet(vertexFile)

        motionFile = motionFormat.format(frameId)
        motion = torch.FloatTensor(np.loadtxt(motionFile)) if pose_ext == '.txt' else tensorIO.ReadTensorFromBinaryFile(motionFile).float()
        motion = motion.expand(1, -1)

        verts = v.expand(1, -1, 3)
        verts_unposed = lbs_fn.unpose(motion, scale, verts / global_scaling)
        outputFile = outputFormat.format(sequence, frameId)
        imageInterpolater.saveplymesh(outputFile, verts_unposed[0], f)