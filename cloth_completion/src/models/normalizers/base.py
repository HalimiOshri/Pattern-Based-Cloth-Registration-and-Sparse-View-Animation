import numpy as np
import torch.nn as nn
from models.aux.ae_geometry import GeometryBase
from src.rendering.render import sample_uv, values_to_uv
import torch

# Responsible of coordinates normalization from vertices to normalized uv coordinates
class Normalizer(nn.Module):
    '''
            Responsible to convert mesh geometry and texture signal to a normalized uv signal
            (on which a core network can operate)
            The normalize should provide API for the functions normalize(), unormalize().
            To support a dataset with another interface, override parse().
            To support a different normalization scheme, override normalize(), unormalize().
    '''
    def __init__(self, dataset, kinematic_model_conf):
        super(Normalizer, self).__init__()
        self.parse(dataset, kinematic_model_conf)

    # TODO: why when we normalize we do this w.r.t the unposed mean, but when we reverse the normalization we do this w.r.t cloth template?
    def normalize(self, unnormalized, return_texture=False):
        pose = unnormalized["pose"]
        verts = unnormalized["verts"]
        tex = unnormalized["tex"]

        scale = self.geometry_base.lbs_scale.expand(pose.shape[0], -1)
        unposed_verts = self.geometry_base.lbs_fn.unpose(pose, scale, verts / self.geometry_base.global_scaling)
        unposed_verts_delta = unposed_verts - self.verts_unposed_mean
        verts_normalized_uv = values_to_uv(unposed_verts_delta, self.geometry_base.index_image,
                                                self.geometry_base.bary_image)
        tex_normalized = (tex - self.tex_mean) / self.tex_std if return_texture else False

        return {"verts_normalized_uv": verts_normalized_uv, "tex_normalized": tex_normalized}

    def unnormalize(self, normalized, return_texture=False):
        pose = normalized["pose"]
        verts_normalized_uv = normalized["verts_normalized_uv"]
        tex_normalized = normalized["tex_normalized"]

        unposed_verts_delta = sample_uv(verts_normalized_uv, self.geometry_base.uv_coords,
                                            self.geometry_base.uv_mapping)
        unposed_verts = self.geometry_base.lbs_template_verts + unposed_verts_delta
        scale = self.geometry_base.lbs_scale.expand(pose.shape[0], -1)
        verts = (
                self.geometry_base.lbs_fn(pose, scale, unposed_verts)
                * self.geometry_base.global_scaling[np.newaxis]
        )

        tex = tex_normalized * self.tex_std + self.tex_mean if return_texture else None

        return {"verts": verts, "tex": tex}

    def parse(self, dataset, kinematic_model_conf):
        # responsible for lbs
        self.geometry_base = GeometryBase(
            uv_size=dataset.uv_size,
            lbs_scale=dataset.clothes_lbs_scale,
            lbs_template_verts=dataset.clothes_lbs_template_verts,
            uv_coords=dataset.clothes_uv_coords,
            uv_mapping=dataset.clothes_uv_mapping,
            uv_faces=dataset.clothes_uv_faces,
            nbs_idxs=dataset.clothes_nbs_idxs,
            nbs_weights=dataset.clothes_nbs_weights,
            global_scaling=dataset.global_scaling,
            lbs=kinematic_model_conf["clothes_lbs"],
        )