from models.normalizers.base import Normalizer

class PartialInputNormalizer(Normalizer):
    def normalize(self, unnormalized, return_texture=False):
        normalized = super(PartialInputNormalizer, self).normalize(unnormalized)
        partial_mask = unnormalized["partial_mask"]

        partial_mask = 1 * (partial_mask > 0)
        verts_normalized_uv = partial_mask * normalized["verts_normalized_uv"]
        tex_normalized = partial_mask * normalized["tex_normalized"]

        return {"verts_normalized_uv": verts_normalized_uv, "tex_normalized": tex_normalized}