import numpy as np
import moderngl


def _to_glsl_type(dtype, num_channels=3):
    if num_channels not in [2, 3, 4]:
        raise ValueError("Unsupported number of channels")

    if dtype == np.float32:
        return f"vec{num_channels}"
    elif dtype == np.int32:
        return f"vec{num_channels}"
    elif dtype == np.uint8:
        return f"uvec{num_channels}"
    else:
        raise ValueError("Unsupported dtype")


def _to_layout(dtype, num_channels):
    if dtype == np.float32:
        return f"{num_channels}f4"
    elif dtype == np.int32:
        return f"{num_channels}i4"
    elif dtype == np.uint8:
        return f"{num_channels}u1"
    else:
        raise ValueError("Unsupported dtype")


def K_to_pm(K, height, width, near_clip, far_clip):
    q = np.array(
        [
            [
                2.0 * K[0][0] / width,
                2.0 * K[0][1] / width,
                (2.0 * K[0][2] / width) - 1.0,
                0.0,
            ],
            [0.0, -(2.0 * K[1][1] / height), -((2.0 * K[1][2] / height) - 1.0), 0.0],
            [
                0.0,
                0.0,
                (far_clip + near_clip) / (far_clip - near_clip),
                -2.0 * far_clip * near_clip / (far_clip - near_clip),
            ],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    return q.T.reshape((-1,))


def RT_to_mv(RT):
    mv = np.zeros((4, 4), dtype=np.float32)
    mv[:4, :3] = RT.T
    mv[3, 3] = 1.0
    return mv.reshape((-1,))


class UVRenderer:
    def __init__(
        self, uv_size, uv_coords, uv_faces, num_channels=3, dtype=np.float32, ctx=None
    ):

        self.ctx = (
            moderngl.create_context(standalone=True, backend="egl")
            if ctx is None
            else ctx
        )

        self.uv_size = uv_size
        self.uv_coords = uv_coords
        self.uv_faces = uv_faces
        self.num_channels = num_channels

        self._glsl_type = _to_glsl_type(dtype, num_channels)
        self._layout = _to_layout(dtype, num_channels)
        self._dtype_str = self._layout[1:]

        self.program = self.ctx.program(
            vertex_shader=(
                f"""
                #version 330
                in vec2 uv_coords;
                in {self._glsl_type} values;
                out {self._glsl_type} values_frag;
                """
                """
                void main() {
                    values_frag = values;
                    gl_Position.x = 2.0 * uv_coords.x - 1.0;
                    gl_Position.y = 2.0 * uv_coords.y - 1.0;
                    gl_Position.z = 0.5;
                    gl_Position.w = 1.0;
                }
            """
            ),
            fragment_shader=(
                f"""
                #version 330
                in {self._glsl_type} values_frag;

                layout(location = 0) out {self._glsl_type} out_values;
                """
                """
                void main() {
                    out_values = values_frag;
                }
            """
            ),
        )

        self.uv_coords_buffer = self.ctx.buffer(uv_coords.reshape((-1,)).tobytes())
        self.uv_faces_buffer = self.ctx.buffer(uv_faces.reshape((-1,)).tobytes())

        self.fbo = self.ctx.simple_framebuffer(
            (self.uv_size, self.uv_size),
            dtype=self._dtype_str,
            components=self.num_channels,
        )

    def render(self, values):

        self.fbo.use()
        self.ctx.clear(viewport=(self.uv_size, self.uv_size))

        values_buffer = self.ctx.buffer(values.reshape((-1,)).tobytes())

        vao = self.ctx.vertex_array(
            self.program,
            [
                self.uv_coords_buffer.bind("uv_coords", layout="2f"),
                values_buffer.bind("values", layout=self._layout),
            ],
            self.uv_faces_buffer,
        )

        vao.render()

        rendered_bytes = self.fbo.read(
            components=self.num_channels, dtype=self._dtype_str
        )
        return np.frombuffer(rendered_bytes, dtype=self._dtype_str).reshape(
            (self.uv_size, self.uv_size, self.num_channels)
        )


class VertexRenderer:
    def __init__(
        self, faces, height, width, num_channels=3, dtype=np.float32, ctx=None
    ):

        self.ctx = (
            moderngl.create_context(standalone=True, backend="egl")
            if ctx is None
            else ctx
        )

        self.faces = faces
        self.height = height
        self.width = width
        self.num_channels = num_channels

        self._glsl_type = _to_glsl_type(dtype, num_channels)
        self._layout = _to_layout(dtype, num_channels)
        self._dtype_str = self._layout[1:]

        self.program = self.ctx.program(
            vertex_shader=(
                f"""
                #version 330

                uniform mat4 p_matrix;
                uniform mat4 mv_matrix;

                in vec3 verts;
                in {self._glsl_type} values;

                out {self._glsl_type} values_frag;
                """
                """
                void main() {
                    vec4 eye_position = mv_matrix * vec4(verts, 1.0);
                    gl_Position = p_matrix * eye_position;
                    values_frag = values;
                }
            """
            ),
            fragment_shader=(
                f"""
                #version 330

                in {self._glsl_type} values_frag;

                layout(location = 0) out {self._glsl_type} out_values;
                """
                """
                void main() {
                    out_values = values_frag;
                }
            """
            ),
        )

        # TODO: should faces be only here?
        self.faces_buffer = self.ctx.buffer(faces.reshape((-1,)).tobytes())

        self.rbo_values = self.ctx.renderbuffer(
            (width, height), components=self.num_channels, dtype=self._dtype_str
        )
        self.rbo_depth = self.ctx.depth_renderbuffer((width, height))

        self.fbo = self.ctx.framebuffer(
            color_attachments=(self.rbo_values), depth_attachment=(self.rbo_depth)
        )

    def render(self, verts, values, RT, K, near_clip=0.1, near_far=10000.0):

        self.program["p_matrix"] = tuple(
            K_to_pm(K, self.height, self.width, near_clip, near_far)
        )
        self.program["mv_matrix"] = tuple(RT_to_mv(RT))

        self.fbo.use()
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.clear(viewport=(self.width, self.height))

        values_buffer = self.ctx.buffer(values.reshape((-1,)).tobytes())
        verts_buffer = self.ctx.buffer(verts.reshape((-1,)).tobytes())

        vao = self.ctx.vertex_array(
            self.program,
            [
                verts_buffer.bind("verts", layout="3f"),
                values_buffer.bind("values", layout=self._layout),
            ],
            index_buffer=self.faces_buffer,
        )

        vao.render()

        rendered_bytes = self.fbo.read(
            components=self.num_channels, dtype=self._dtype_str
        )
        return np.frombuffer(rendered_bytes, dtype=self._dtype_str).reshape(
            (self.height, self.width, self.num_channels)
        )[::-1]

    def render_batch(self, verts, values, RT, K, near_clip=0.1, near_far=10000.0):
        result = []
        for i in range(verts.shape[0]):
            result.append(
                self.render(
                    verts[i : i + 1], values[i : i + 1], RT, K, near_clip, near_far
                )
            )
        return np.stack(result)
