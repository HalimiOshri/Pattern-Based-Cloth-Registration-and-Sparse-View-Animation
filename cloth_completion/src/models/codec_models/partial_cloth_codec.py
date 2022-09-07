from models.codec_models.cloth_codec import CodecClothModel
from models.normalizers.partial_input import PartialInputNormalizer

class CodecPartialClothModel(CodecClothModel):
    def build_normalizer(self, dataset, kinematic_model_conf):
        return PartialInputNormalizer(dataset, kinematic_model_conf)

    def parse_input(self, motion, clothes_verts, clothes_tex, **kwargs):
        unnormalized = {"pose": motion, "verts": clothes_verts, "tex": clothes_tex, "partial_mask": kwargs["partial_mask"]}
        return unnormalized