import os
import torch
import numpy as np
import imageInterpolater
from src.pyutils import tensorIO
from src.kinematic_modeling.lbs.linear_blend_skinning import LinearBlendSkinning

sequence = 's--20210823--1323--0000000--pilot--patternCloth'
template_path = '/mnt/home/oshrihalimi/pycharm/cloth_completion/assets/clothes_unposed_mean.ply'
output_dir = '/mnt/home/oshrihalimi/cloth_completion/data_processing/{}/posedLBS/'.format(sequence)
outputFormat = '/mnt/home/oshrihalimi/cloth_completion/data_processing/{}/posedLBS/{:06d}.ply'
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

if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    lbs_fn = LinearBlendSkinning(template_path=template_path,
                                 model_json_path=lbs["model_json_path"], model_txt_path=lbs["model_txt_path"], num_max_skin_joints=lbs["num_max_skin_joints"])

    with open(lbs["scale_path"]) as f:
        scale = torch.Tensor([float(x) for x in f.read().splitlines()]).expand(1, -1)
    global_scaling = torch.Tensor([[10., -10., -10.]])
    pose_ext = os.path.splitext(motionFormat)[1]

    template_save_path = '/mnt/home/oshrihalimi/cloth_completion/data_processing/{}/posedLBS/{}.ply'
    imageInterpolater.saveplymesh(template_save_path.format(sequence, 'template'), lbs_fn.mesh_vertices, lbs_fn.mesh_faces)
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

        template_posed = lbs_fn(motion, scale, templmesh=lbs_fn.template_verts[None, :, :]) * torch.tensor([10.,-10.,-10.])[None, None, :]
        outputFile = outputFormat.format(sequence, frameId)
        imageInterpolater.saveplymesh(outputFile, template_posed[0], f)