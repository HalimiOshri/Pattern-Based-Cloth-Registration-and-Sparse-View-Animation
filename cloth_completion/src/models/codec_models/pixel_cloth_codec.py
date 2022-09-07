from models.codec_models.cloth_codec import CodecClothModel
from models.normalizers.pixel_input import PixelInputNormalizer

class PixelCodecClothModel(CodecClothModel):
    def build_normalizer(self, dataset, kinematic_model_conf):
        return PixelInputNormalizer(dataset, kinematic_model_conf)

    def parse_input(self, **inputs):
        unnormalized = {"per_camera_delta_pixel_uv": inputs["per_camera_delta_pixel_uv"]}
        return unnormalized

    def parse_final(self, inputs, net_output):
        normalized_output = {"verts_normalized_uv": net_output["verts_uv_delta"], "posed_LBS_verts": inputs["posed_LBS_verts"]}
        return normalized_output

    def make_preds(self, output, inputs):
        preds = dict()
        preds.update({"clothes_verts": output["verts"], "clothes_tex": output["tex"]})
        return preds