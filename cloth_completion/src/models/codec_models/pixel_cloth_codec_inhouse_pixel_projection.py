from models.codec_models.pixel_cloth_codec import PixelCodecClothModel
from models.normalizers.pixel_input import InhousePixelProjectionPixelPixelInputNormalizer

class InhousePixelProjectionPixelCodecClothModel(PixelCodecClothModel):
    def build_normalizer(self, dataset, kinematic_model_conf):
        return InhousePixelProjectionPixelPixelInputNormalizer(dataset, kinematic_model_conf)

    def parse_input(self, **inputs):
        unnormalized = {"per_camera_detection_pixel_uv": inputs["per_camera_delta_pixel_uv"], "posed_kinematic_verts": inputs["posed_LBS_verts"], "cloth_faces": self.clothes_faces}
        return unnormalized

    def parse_final(self, inputs, net_output):
        # we pass only the present frame of the posed kinematic model, on which the offset predicted by the network is applied
        normalized_output = {"verts_normalized_uv": net_output["verts_uv_delta"], "posed_LBS_verts": net_output["deformed_kinematic_mesh"][:, 0, ...]}
        return normalized_output