import torch
import torch.nn as nn
from src.geometry_layers.normals import fnrmls
from models.normalizers.base import Normalizer
from models.aux.io import name_to_class

# interface class between the network's architecture and the training procedure:
# 1) generates the network given model configuration and dataset
# 2) output the model predictions and the loss terms

class CodecClothModel(nn.Module):
    '''
    Base class for CodecClothModel.
    To implement a different architecture, override build_net().
    To implement a different normalization, override build_normalizer().
    '''
    def __init__(self, dataset, **model_conf):
        super().__init__()
        print(model_conf)
        self.normalizer = self.build_normalizer(dataset, model_conf["kinematic_model"])
        self.normal_net = self.build_net(model_conf["net"])

        # TODO: build mesh data structure
        self.clothes_faces = dataset.clothes_faces
        self.nbs_idxs = dataset.clothes_nbs_idxs
        self.nbs_weights = dataset.clothes_nbs_weights


    def build_net(self, net_conf):
        return name_to_class(net_conf["class_name"])(**net_conf)

    def build_normalizer(self, dataset, kinematic_model_conf):
        '''
        The normalize should provide API for the functions normalize(), unormalize()
        :param dataset:
        :return:
        '''
        return Normalizer(dataset, kinematic_model_conf)

    def forward(self, **inputs):
        parsed_input = self.parse_input(**inputs)
        normalized_input = self.normalizer.normalize(parsed_input)

        net_output = self.normal_net(normalized_input)

        parsed_final = self.parse_final(inputs, net_output)
        output = self.normalizer.unnormalize(parsed_final)

        preds = self.make_preds(output, inputs)
        preds = self.calc_loss_terms(preds, inputs) # TODO: find a more general solution to pass the target signal

        return preds

    def make_preds(self, output, inputs):
        preds = dict()
        preds.update({"clothes_verts": output["verts"], "clothes_tex": output["tex"], "kinematic_uv_input": output["kinematic_uv_input"]})
        return preds

    def parse_input(self, **inputs):
        unnormalized = {"pose": inputs["motion"], "verts": inputs["clothes_verts"], "tex": inputs["clothes_tex"]}
        return unnormalized

    def parse_final(self, inputs, net_output):
        normalized_output = {"pose": inputs["motion"], "verts_normalized_uv": net_output["verts_uv_delta"], "tex_normalized": net_output["tex_mean_rec"]}
        return normalized_output

    def calc_loss_terms(self, preds, inputs):
        clothes_verts = inputs["clothes_verts"]

        # clothes_tex = self.decoder.resize_tex(clothes_tex)
        # clothes_tex_norm = (clothes_tex - self.decoder.clothes_tex_mean) / self.decoder.clothes_tex_std

        loss_clothes_verts_rec = (preds["clothes_verts"] - clothes_verts).pow(2).mean(dim=(1, 2))

        clothes_normals_predicted = fnrmls(preds["clothes_verts"], torch.Tensor(self.clothes_faces).to(dtype=torch.int64))
        clothes_normals_gt = fnrmls(clothes_verts, torch.Tensor(self.clothes_faces).to(dtype=torch.int64))
        loss_clothes_normals_rec = (clothes_normals_predicted - clothes_normals_gt).pow(2).mean(dim=(1, 2))

        # resized_clothes_tex_mask = self.decoder.resize_tex(kwargs['clothes_tex_mask'])
        # loss_clothes_tex_rec = (
        #         resized_clothes_tex_mask
        #         * (preds["clothes_tex_norm"] - clothes_tex_norm) +
        #         ((1.0 - resized_clothes_tex_mask) * preds["clothes_tex_norm"] if self.invisible_mean else 0.0)
        # ).abs().mean(dim=(1, 2, 3))

        # TODO: add laplacian
        # loss_clothes_verts_laplacian = laplacian_loss(preds["clothes_verts"], clothes_verts, self.nbs_idxs, self.nbs_weights)

        # computing normal average losses
        preds.update(
            loss_clothes_verts_rec=loss_clothes_verts_rec,
            loss_clothes_verts_laplacian=0.0, #loss_clothes_verts_laplacian,
            loss_clothes_tex_rec=0.0, #loss_clothes_tex_rec,
            loss_clothes_normals_rec=loss_clothes_normals_rec,
        )

        return preds


class TotalLoss(nn.Module):
    def __init__(self, weights, kl_type="default", **kwargs):
        super().__init__()
        self.weights = weights
        self.kl_type = kl_type
        assert self.kl_type in ["default", "anneal"]

    def forward(self, preds, targets, inputs=None, iteration=None):

        loss_dict = dict()

        loss_dict.update(
            loss_clothes_verts_rec=preds["loss_clothes_verts_rec"].mean(),
            loss_clothes_normals_rec=preds["loss_clothes_normals_rec"].mean(),
            # loss_clothes_tex_rec=preds["loss_clothes_tex_rec"].mean(),
            # loss_clothes_verts_laplacian=preds["loss_clothes_verts_laplacian"].mean(),
            # loss_kl_embs=kl_loss(preds["embs_mu"], preds["embs_logvar"]),
        )

        loss = (loss_dict["loss_clothes_verts_rec"] * self.weights.clothes_geometry_rec
                + loss_dict["loss_clothes_normals_rec"] * self.weights.loss_clothes_normals_rec
                # + loss_dict["loss_clothes_tex_rec"] * self.weights.clothes_tex_rec
                # + loss_dict["loss_clothes_verts_laplacian"] * self.weights.clothes_geometry_laplacian
                )

        # if self.kl_type == "default":
        #     # standard KL
        #     loss += loss_dict["loss_kl_embs"] * self.weights.kl
        # elif self.kl_type == "anneal" and iteration is not None:
        #     c = self.weights.kl_anneal
        #     kl_weight = (1.0 - min(iteration, c.end_at) / c.end_at) * (
        #             c.initial_value - c.min_value
        #     ) + c.min_value
        #     loss += loss_dict["loss_kl_embs"] * kl_weight
        #     loss_dict["loss_kl_weight"] = torch.tensor(kl_weight)

        loss_dict.update(loss_total=loss)

        return loss, loss_dict

