import sys
import os
import torch as th
import numpy as np



###################
## He Wen calibration file parsing
def parse_KRT(lines):
    K = np.array([lines[0].split(), lines[1].split(), lines[2].split()], dtype=np.float32)
    # line 3 is distortion
    Rt = np.array([lines[4].split(), lines[5].split(), lines[6].split()], dtype=np.float32)
    return K, Rt

def read_KRT(lines):
    KRT_ = dict()
    K_ = dict()
    Rt_ = dict()
    u = 0
    while u < len(lines):
        cam = lines[u].strip()
        K, Rt = parse_KRT(lines[u+1:u+8])
        K_[cam] = K
        Rt_[cam] = Rt
        u += 9
        #t = np.matmul(-np.linalg.inv(Rt[:, :3]), Rt[:, 3])
        KRT_[cam] = np.dot(K, Rt)
    return KRT_,K_,Rt_
####################

class Camera:
    def __init__(self,pos,rot,focal,princpt):
        self.pos = pos
        self.rot = rot
        self.focal = focal
        self.princpt = princpt
    def to_cuda(self):
        self.pos = self.pos.cuda()
        self.rot = self.rot.cuda()
        self.focal = self.focal.cuda()
        self.princpt = self.princpt.cuda()


def loadCameraSetInfo(calibrationFile,cameraIndicesFile,cameraImageWidth,cameraImageHeight,renderingScale,to_cuda = True):
    if not os.path.exists(calibrationFile):
        print(f'KRT file {calibrationFile} not found!')
        exit()
    with open(calibrationFile) as f:
        lines = [l for l in f]
    KRT,K,Rt = read_KRT(lines)

    cameraIndices = []
    if not os.path.exists(cameraIndicesFile):
        print(f'Camera index file {cameraIndicesFile} not found!')
        exit()
    else:
        cameraIndices = open(cameraIndicesFile, "r").read().split()
        cameraIndices = list(map(int, cameraIndices))


    K_focal_list = []
    K_princpt_list = []
    K_list = []
    camera_rotation_list = []
    camera_tranlsation_list = []
    target_image_list = []
    cam_indices = []
    num_cameras = 0
    #Rt_list = []
    for cam in cameraIndices:
        if str(cam) in K.keys() and str(cam) in Rt.keys():
            K_focal_list.append(K[str(cam)][0:2,0:2])
            K_princpt_list.append(K[str(cam)][0:2,2])
            K_list.append(K[str(cam)])
            
            camera_rotation_list.append(Rt[str(cam)][:,0:3])
            camera_tranlsation_list.append(Rt[str(cam)][:,3])

            cam_indices.append(cam)

            num_cameras += 1

    cam_focal = th.from_numpy(np.asarray(K_focal_list))
    cam_princpt = th.from_numpy(np.asarray(K_princpt_list))
    cam_intrinsic = th.from_numpy(np.asarray(K_list))
    w = int(cameraImageWidth * renderingScale)
    h = int(cameraImageHeight * renderingScale)

    scaling_w = float(w) / float(cameraImageWidth)
    scaling_h = float(h) / float(cameraImageHeight)

    cam_focal[:,0,:] *= scaling_w
    cam_focal[:,1,:] *= scaling_h

    cam_princpt[:,0] *= scaling_w
    cam_princpt[:,1] *= scaling_h

    cam_intrinsic[:,0] *= scaling_w
    cam_intrinsic[:,1] *= scaling_h

    camera_rotation = th.from_numpy(np.asarray(camera_rotation_list))
    camera_translation = th.from_numpy(np.asarray(camera_tranlsation_list))
    camera_position = -th.matmul(camera_rotation.inverse(),camera_translation.unsqueeze(2)).squeeze()

    if to_cuda:
        cam_focal = cam_focal.cuda()
        cam_princpt = cam_princpt.cuda()
        camera_rotation =camera_rotation.cuda()
        camera_position = camera_position.cuda()
        cam_intrinsic = cam_intrinsic.cuda()
        camera_translation = camera_translation.cuda()

    return [cam_focal,cam_princpt,cam_intrinsic,camera_rotation,camera_translation,camera_position,w,h,cam_indices]


def loadCameraROIInfo(calibrationFile,cameraIndicesFile,roiCoords,w,h,to_cuda = True):
    if not os.path.exists(calibrationFile):
        print(f'KRT file {calibrationFile} not found!')
        exit()
    with open(calibrationFile) as f:
        lines = [l for l in f]
    KRT,K,Rt = read_KRT(lines)

    cameraIndices = []
    if not os.path.exists(cameraIndicesFile):
        print(f'Camera index file {cameraIndicesFile} not found!')
        exit()
    else:
        cameraIndices = open(cameraIndicesFile, "r").read().split()
        cameraIndices = list(map(int, cameraIndices))


    K_focal_list = []
    K_princpt_list = []
    K_list = []
    camera_rotation_list = []
    camera_tranlsation_list = []
    target_image_list = []
    cam_indices = []
    num_cameras = 0
    #Rt_list = []
    for cam in cameraIndices:
        if str(cam) in K.keys() and str(cam) in Rt.keys():
            K_focal_list.append(K[str(cam)][0:2,0:2])
            K_princpt_list.append(K[str(cam)][0:2,2])
            K_list.append(K[str(cam)])
            
            camera_rotation_list.append(Rt[str(cam)][:,0:3])
            camera_tranlsation_list.append(Rt[str(cam)][:,3])

            cam_indices.append(cam)

            num_cameras += 1
    
    print("Num cameras: %d" % num_cameras)

    
    cam_focal = th.from_numpy(np.asarray(K_focal_list)).unsqueeze(0)
    cam_princpt = th.from_numpy(np.asarray(K_princpt_list)).unsqueeze(0)

    #print("Rendering resolution: %d x %d" % (w,h))
    renderSize = th.tensor([float(w),float(h)])

    # segments x cameras x 2
    roiSize = (roiCoords[:,:,1] - roiCoords[:,:,0]).float()
    
    # segments x cameras x 2
    scaling = renderSize / roiSize

    # segments x cameras x 2 x 2
    roi_cam_focal = cam_focal * scaling.unsqueeze(3)
    roi_cam_focal = roi_cam_focal.permute(1,0,2,3)

    # segments x cameras x 2
    roi_cam_princpt = (cam_princpt - roiCoords[:,:,0]) * scaling
    roi_cam_princpt = roi_cam_princpt.permute(1,0,2)

    if to_cuda:
        roi_cam_focal = roi_cam_focal.cuda()
        roi_cam_princpt = roi_cam_princpt.cuda()

    return [roi_cam_focal,roi_cam_princpt]
