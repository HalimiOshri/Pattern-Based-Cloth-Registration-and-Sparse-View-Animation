import numpy as np

import torch
import torch as th
import torch.nn as nn
import torch.utils.data

import cv2


def label_image(
    image,
    label,
    font_scale=1.0,
    font_thickness=1,
    label_origin=(10, 64),
    font_color=(255, 255, 255),
    font=cv2.FONT_HERSHEY_SIMPLEX,
):
    text_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    image[
        label_origin[1] - text_size[1] : label_origin[1] + baseline,
        label_origin[0] : label_origin[0] + text_size[0],
    ] = (255 - font_color[0], 255 - font_color[1], 255 - font_color[2])
    cv2.putText(
        image, label, label_origin, font, font_scale, font_color, font_thickness
    )
    return image




def linear2displayBatchGray(val): # 3 height width
    gamma = 2.0
    #np_scal = np.array([2.0, 0.95, 1.05],dtype=np.float32)
    np_scal = np.array([1.33, 1.33, 1.33],dtype=np.float32)
    scaling = torch.from_numpy(np_scal).to(val.device)
    val = val.float()/255.0 * scaling[None,:,None,None]
    return (torch.clamp(val,0,1)**(1.0/gamma)) * 255.

def linear2displayBatch_v0(val): # 3 height width
    gamma = 2.0
    #np_scal = np.array([2.0, 0.95, 1.05],dtype=np.float32)
    np_scal = np.array([1.05, 0.95, 2.0],dtype=np.float32)
    scaling = torch.from_numpy(np_scal).to(val.device)
    val = val.float()/255.0 * scaling[None,:,None,None]
    return (torch.clamp(val,0,1)**(1.0/gamma)) * 255.

def linear2display(val):  # 3 height width
    gamma = 2.0
    # np_scal = np.array([2.0, 0.95, 1.05],dtype=np.float32)
    np_scal = np.array([1.05, 0.95, 2.0], dtype=np.float32)
    scaling = torch.from_numpy(np_scal).to(val.device)
    val = val.float() / 255.0 * scaling[:, None, None]
    return (torch.clamp(val, 0, 1) ** (1.0 / gamma)) * 255.0

def linear2srgb(img, gamma=2.4):
    linear_part = img * 12.92
    exp_part = 1.055 * (img ** (1 / gamma)) - 0.055
    if isinstance(img, torch.Tensor):
        return torch.where(img <= 0.0031308, linear_part, exp_part)
    else:
        return np.where(img <= 0.0031308, linear_part, exp_part)



def linear2displayBatch(val, gamma=1.5, wbscale=np.array([1.05, 0.95, 1.45],dtype=np.float32), black=5.0/255.0, mode='srgb'):
    scaling = torch.from_numpy(wbscale).to(val.device)
    val = val.float()/255.0 * scaling[None,:,None,None]-black
    if mode=='srgb':
        val = linear2srgb(val, gamma=gamma)
    else:
        val = val**(1.0/gamma)
    return torch.clamp(val,0,1)* 255.



def index_selection_nd(x, I, dim):
    target_shape = [*x.shape]
    del target_shape[dim]
    target_shape[dim:dim] = [*I.shape]
    return x.index_select(dim, I.view(-1)).reshape(target_shape)


def computeLaplacian(M, neiboridx, neiborwts):  # M is B N 3
    numnei = neiboridx.size(1)
    lapval = index_selection_nd(M, neiboridx, 1) * neiborwts.unsqueeze(2).unsqueeze(0)
    return lapval.sum(2) + M


# t_mesh_verts: B * V * 3
# t_skinning_wts: J * V
# out: weighted average of vertices for each joint: B * J * 3
def computeLookAtPoints(t_mesh_verts, t_skinning_wts):
    lookatpt = (t_skinning_wts[None, :, :, None] * t_mesh_verts[:, None, ...]).sum(
        2
    ) / (
        1e-6 + t_skinning_wts.sum(1)[None, :, None]
    )  # N J 3
    return lookatpt


def computeNormal(geometry_in):
    # nbatch = geometry_in.size(0)
    # height = geometry_in.size(1)
    # width = geometry_in.size(2)
    normal = (geometry_in[...,0,:] - geometry_in[...,4,:]).cross(geometry_in[...,1,:] - geometry_in[...,4,:], dim = 3)
    normal = normal + (geometry_in[...,1,:] - geometry_in[...,4,:]).cross(geometry_in[...,2,:] - geometry_in[...,4,:], dim = 3)
    normal = normal + (geometry_in[...,2,:] - geometry_in[...,4,:]).cross(geometry_in[...,3,:] - geometry_in[...,4,:], dim = 3)
    normal = normal + (geometry_in[...,3,:] - geometry_in[...,4,:]).cross(geometry_in[...,0,:] - geometry_in[...,4,:], dim = 3)
    normal = normal / torch.clamp(normal.pow(2).sum(3, keepdim = True).sqrt(), min=1e-6)
    return normal.permute(0,3,1,2)



class NormalComputer(nn.Module):
    def __init__(self, height, width, maskin = None):
        super(NormalComputer, self).__init__()
        # self.register_buffer('eye', (torch.eye(3)).to(device))        
        # self.register_buffer('zero', (torch.zeros(1,)).to(device))  

        patchttnum = 5 #neighbor + self
        patchmatch_uvpos = np.zeros((height,width,patchttnum,2),dtype=np.int32)
        vec_standuv = np.indices((height,width)).swapaxes(0,2).swapaxes(0,1).astype(np.int32).reshape(height,width,1,2)
        patchmatch_uvpos = patchmatch_uvpos + vec_standuv
        localpatchcoord = np.zeros((patchttnum,2),dtype=np.int32)
        localpatchcoord = np.array([[-1,0],[0,1],[1,0],[0,-1],[0,0]]).astype(np.int32)

        patchmatch_uvpos =  patchmatch_uvpos + localpatchcoord.reshape(1,1,patchttnum,2)
        patchmatch_uvpos[...,0] = np.clip(patchmatch_uvpos[...,0],0,height-1)
        patchmatch_uvpos[...,1] = np.clip(patchmatch_uvpos[...,1],0,width-1)

        #geoemtry mask , apply simiilar to texture mask
        #mesh_mask_int = mesh_mask.reshape(height,width).astype(np.int32)
        if maskin is None:
            maskin = np.ones((height,width),dtype=np.int32)
        mesh_mask_int = maskin.reshape(height,width).astype(np.int32) #using all pixel valid mask; can use a tailored mask
        patchmatch_mask = mesh_mask_int[patchmatch_uvpos[...,0],patchmatch_uvpos[...,1]].reshape(height,width,patchttnum,1)
        patch_indicemap = patchmatch_uvpos * patchmatch_mask + (1 - patchmatch_mask) * vec_standuv

        tensor_patch_geoindicemap = torch.from_numpy(patch_indicemap).long()
        tensor_patch_geoindicemap1d = tensor_patch_geoindicemap[...,0] * width + tensor_patch_geoindicemap[...,1]

        self.register_buffer('tensor_patch_geoindicemap1d', tensor_patch_geoindicemap1d)  
        # tensor_patchmatch_uvpos = torch.from_numpy(patchmatch_uvpos).long()
        # tensor_vec_standuv = torch.from_numpy(vec_standuv).long()
    
    def forward(self, t_georecon): # in: N 3 H W
        #pdb.set_trace()
        geometry_in = index_selection_nd(t_georecon.view(t_georecon.size(0), t_georecon.size(1), -1), self.tensor_patch_geoindicemap1d, 2).permute(0,2,3,4,1)         
        normal = (geometry_in[...,0,:] - geometry_in[...,4,:]).cross(geometry_in[...,1,:] - geometry_in[...,4,:], dim = 3)
        normal = normal + (geometry_in[...,1,:] - geometry_in[...,4,:]).cross(geometry_in[...,2,:] - geometry_in[...,4,:], dim = 3)
        normal = normal + (geometry_in[...,2,:] - geometry_in[...,4,:]).cross(geometry_in[...,3,:] - geometry_in[...,4,:], dim = 3)
        normal = normal + (geometry_in[...,3,:] - geometry_in[...,4,:]).cross(geometry_in[...,0,:] - geometry_in[...,4,:], dim = 3)
        #normal = normal / torch.clamp(normal.pow(2).sum(3, keepdim = True).sqrt(), min=1e-6)
        normal = normal / ((normal.pow(2).sum(3, keepdim = True).sqrt() + 1e-6).detach())
        return normal.permute(0,3,1,2)