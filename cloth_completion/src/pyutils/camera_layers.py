import numpy as np

import torch
import cv2 as cv

def index_selection_nd(x, I, dim):
    target_shape = [*x.shape]
    del target_shape[dim]
    target_shape[dim:dim] = [*I.shape]
    return x.index_select(dim, I.view(-1)).reshape(target_shape)



  ##################################

class RotateCameraLayerModule(torch.nn.Module):
    def __init__(self,intrinsics3x3, extrinsics3x4, zfactor = 1.0): # assuming to be all tensor input
        super(RotateCameraLayerModule, self).__init__()        

        self.register_buffer("intrinsics", intrinsics3x3)
        extrinsics3x3 = extrinsics3x4[:3,:3]
        camctr = -extrinsics3x3.t().mv(extrinsics3x4[:3,3])

        self.register_buffer("extrotmat", extrinsics3x3)
        self.register_buffer("camctr", camctr)
        self.register_buffer("extrinsics", extrinsics3x4)

        self.rotspeed = 0.25
        self.zoomfactor = zfactor


    def getCamCenter(self):        
        return self.camctr

    def changeOpticalDist(self,rotctr, scale):
        tmpdir = self.camctr - rotctr
        tmppos = rotctr + tmpdir * scale
        retextmat = self.extrinsics
        retextmat[:3,3] = -self.extrotmat.mv(tmppos)
        return retextmat



    def forward(self, rotaxle, rotctr, incremental, scaling = 1.0): # assuming all tensor input

        dr = rotaxle * (np.pi / 180 * self.rotspeed * incremental)        
        dR, _ = cv.Rodrigues(dr.numpy())
        dR = torch.from_numpy(dR)
        # print(dR)
        dvec = rotctr -dR.mv(rotctr)
        # print(dvec)
        #print(transfmat.squeeze(0).numpy())        
        t_dRot = torch.eye(4).float()
        t_dRot[:3,:3] = dR.float()
        t_dRot[:3,3] = dvec.float()
        # print(t_dRot)
        #t_dRot_cuda = Variable(t_dRot.cuda()).unsqueeze(0)      
        # print(self.extrinsics)
        
        #curextmat = self.extrinsics
        curextmat = self.changeOpticalDist(rotctr,scaling)
        tensor_extrinsics3x4 = curextmat.mm(t_dRot.data)
        # print(tensor_extrinsics3x4)

        # t_view =  -tensor_extrinsics3x4[0,:3,:3].t().mv(tensor_extrinsics3x4[0,:3,3]) - t_transfmat.data[0,:3,3]
        # t_view /= (t_view.norm()+1e-12)
        # t_view = Variable(t_view.unsqueeze(0).float())

        # tensor_intrinsics3x3 = torch.eye(3).float()
        tensor_intrinsics3x3 = self.intrinsics
        tensor_intrinsics3x3[:2,:2] = self.intrinsics[:2,:2] * self.zoomfactor
        # tensor_intrinsics3x3[:2,2] = self.intrinsics[:2,2]
        
        return tensor_intrinsics3x3, tensor_extrinsics3x4




   