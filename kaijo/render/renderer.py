# SPDX-License-Identifier: GPL-3.0-or-later
"""OpenGL 3.3 core renderer.

One renderer instance serves the main view, the preview thumbnails and the
image export: everything is drawn through the same code path into either
the GTK-provided framebuffer or an offscreen MSAA target, which guarantees
that previews and exported images look identical (up to resolution).
"""

import ctypes

import numpy as np
from OpenGL import GL

from . import meshes
from .camera import quat_to_mat3

# 2D stroke glyphs for the axis labels, in [-1, 1] box units
_GLYPHS = {
    "x": [(-1, -1, 1, 1), (-1, 1, 1, -1)],
    "y": [(-1, 1, 0, 0), (1, 1, 0, 0), (0, 0, 0, -1.2)],
    "z": [(-1, 1, 1, 1), (1, 1, -1, -1), (-1, -1, 1, -1)],
}

_AXES = [
    (np.array([1.0, 0.0, 0.0]), (0.80, 0.15, 0.15), "x"),
    (np.array([0.0, 1.0, 0.0]), (0.10, 0.55, 0.10), "y"),
    (np.array([0.0, 0.0, 1.0]), (0.15, 0.30, 0.85), "z"),
]

_VERT_INSTANCED = """
#version 330 core
layout(location=0) in vec3 in_pos;
layout(location=1) in vec3 in_norm;
layout(location=2) in vec4 c0;
layout(location=3) in vec4 c1;
layout(location=4) in vec4 c2;
layout(location=5) in vec4 c3;
layout(location=6) in vec4 in_color;
layout(location=7) in vec4 in_scale;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec3 v_norm;
out vec4 v_color;
void main() {
    mat4 model = mat4(c0, c1, c2, c3);
    vec4 w = model * vec4(in_pos, 1.0);
    gl_Position = u_proj * u_view * w;
    vec3 s = max(in_scale.xyz, vec3(1e-8));
    v_norm = mat3(u_view) * (mat3(model) * (in_norm / (s * s)));
    v_color = in_color;
}
"""

_VERT_SURFACE = """
#version 330 core
layout(location=0) in vec3 in_pos;
layout(location=1) in vec3 in_norm;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec4 u_color;
out vec3 v_norm;
out vec4 v_color;
void main() {
    gl_Position = u_proj * u_view * vec4(in_pos, 1.0);
    v_norm = mat3(u_view) * in_norm;
    v_color = u_color;
}
"""

_FRAG_PHONG = """
#version 330 core
in vec3 v_norm;
in vec4 v_color;
uniform float u_twosided;
out vec4 frag;
void main() {
    vec3 n = normalize(v_norm);
    if (u_twosided > 0.5 && !gl_FrontFacing) n = -n;
    vec3 L = normalize(vec3(0.35, 0.45, 1.0));
    float diff = max(dot(n, L), 0.0);
    vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
    float spec = pow(max(dot(n, H), 0.0), 48.0) * 0.35;
    vec3 c = v_color.rgb * (0.32 + 0.68 * diff) + vec3(spec);
    frag = vec4(c, v_color.a);
}
"""

_FRAG_HALO = """
#version 330 core
in vec3 v_norm;
in vec4 v_color;
uniform float u_twosided;
out vec4 frag;
void main() {
    vec3 n = normalize(v_norm);
    float rim = 1.0 - abs(n.z);
    float a = v_color.a * (0.15 + 0.85 * pow(rim, 1.6));
    frag = vec4(v_color.rgb, a);
}
"""

_VERT_SURFACE_VC = """
#version 330 core
layout(location=0) in vec3 in_pos;
layout(location=1) in vec3 in_norm;
layout(location=2) in vec4 in_color;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec3 v_norm;
out vec4 v_color;
void main() {
    gl_Position = u_proj * u_view * vec4(in_pos, 1.0);
    v_norm = mat3(u_view) * in_norm;
    v_color = in_color;
}
"""

_VERT_LABEL = """
#version 330 core
layout(location=0) in vec3 a_anchor;
layout(location=1) in vec2 a_corner;
layout(location=2) in vec2 a_uv;
layout(location=3) in float a_zoff;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec2 v_uv;
void main() {
    vec4 v = u_view * vec4(a_anchor, 1.0);
    v.xy += a_corner;          // billboard in view space
    v.z += a_zoff;             // lift towards the viewer, off the sphere
    gl_Position = u_proj * v;
    v_uv = a_uv;
}
"""

_FRAG_LABEL = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 frag;
void main() {
    vec4 c = texture(u_tex, v_uv);
    if (c.a < 0.05) discard;
    frag = c;                  // premultiplied alpha
}
"""

_VERT_LINES = """
#version 330 core
layout(location=0) in vec3 in_pos;
layout(location=1) in vec3 in_color;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec3 v_color;
void main() {
    gl_Position = u_proj * u_view * vec4(in_pos, 1.0);
    v_color = in_color;
}
"""

_FRAG_LINES = """
#version 330 core
in vec3 v_color;
out vec4 frag;
void main() { frag = vec4(v_color, 1.0); }
"""


def _compile(vs_src, fs_src):
    prog = GL.glCreateProgram()
    for kind, src in ((GL.GL_VERTEX_SHADER, vs_src),
                      (GL.GL_FRAGMENT_SHADER, fs_src)):
        sh = GL.glCreateShader(kind)
        GL.glShaderSource(sh, src)
        GL.glCompileShader(sh)
        if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(sh).decode())
        GL.glAttachShader(prog, sh)
        GL.glDeleteShader(sh)
    GL.glLinkProgram(prog)
    if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
        raise RuntimeError(GL.glGetProgramInfoLog(prog).decode())
    return prog


class _Primitive:
    """A unit mesh with an associated per-draw instance buffer."""

    def __init__(self, verts, norms, faces):
        self.nidx = faces.size
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        data = np.hstack([verts, norms]).astype(np.float32)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data,
                        GL.GL_STATIC_DRAW)
        stride = 24
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, stride,
                                 ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, stride,
                                 ctypes.c_void_p(12))
        ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces,
                        GL.GL_STATIC_DRAW)
        # instance buffer: 24 floats = mat4 + color + scale
        self.ibo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.ibo)
        istride = 24 * 4
        for loc in range(2, 8):
            GL.glEnableVertexAttribArray(loc)
            off = (loc - 2) * 16
            GL.glVertexAttribPointer(loc, 4, GL.GL_FLOAT, False, istride,
                                     ctypes.c_void_p(off))
            GL.glVertexAttribDivisor(loc, 1)
        GL.glBindVertexArray(0)

    def draw(self, instances):
        n = len(instances)
        if n == 0:
            return
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.ibo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, instances.nbytes,
                        np.ascontiguousarray(instances, np.float32),
                        GL.GL_STREAM_DRAW)
        GL.glDrawElementsInstanced(GL.GL_TRIANGLES, self.nidx,
                                   GL.GL_UNSIGNED_INT, None, n)
        GL.glBindVertexArray(0)


class SurfaceGPU:
    """GPU buffers for one isosurface (or polyhedron) mesh."""

    def __init__(self, mesh):
        self.nidx = mesh.faces.size
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        data = np.hstack([mesh.vertices, mesh.normals]).astype(np.float32)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data,
                        GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 24,
                                 ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 24,
                                 ctypes.c_void_p(12))
        self._ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, mesh.faces.nbytes,
                        mesh.faces, GL.GL_STATIC_DRAW)
        GL.glBindVertexArray(0)

    def delete(self):
        GL.glDeleteBuffers(2, [self._vbo, self._ebo])
        GL.glDeleteVertexArrays(1, [self.vao])
        self.nidx = 0

    def draw(self):
        if self.nidx:
            GL.glBindVertexArray(self.vao)
            GL.glDrawElements(GL.GL_TRIANGLES, self.nidx,
                              GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)


class ColoredSurfaceGPU:
    """GPU buffers for a static per-vertex-colored mesh (ESP surfaces)."""

    def __init__(self, mesh, colors):
        self.nidx = mesh.faces.size
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        data = np.hstack([mesh.vertices, mesh.normals,
                          np.asarray(colors, np.float32)]).astype(np.float32)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data,
                        GL.GL_STATIC_DRAW)
        stride = 40  # 3 pos + 3 norm + 4 color floats
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, stride,
                                 ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, stride,
                                 ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, False, stride,
                                 ctypes.c_void_p(24))
        self._ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, mesh.faces.nbytes,
                        mesh.faces, GL.GL_STATIC_DRAW)
        GL.glBindVertexArray(0)

    def delete(self):
        GL.glDeleteBuffers(2, [self._vbo, self._ebo])
        GL.glDeleteVertexArrays(1, [self.vao])
        self.nidx = 0

    def draw(self):
        if self.nidx:
            GL.glBindVertexArray(self.vao)
            GL.glDrawElements(GL.GL_TRIANGLES, self.nidx,
                              GL.GL_UNSIGNED_INT, None)
            GL.glBindVertexArray(0)


class TransparentSurface:
    """Merged, per-vertex-colored mesh whose faces are depth-sorted
    back-to-front on every draw.  This is what makes alpha blending
    correct between the two isosurface lobes (and within folds of a
    single lobe): without global sorting, whichever lobe is drawn last
    would always composite on top regardless of depth."""

    def __init__(self, parts):
        """parts: list of (SurfaceMesh, rgba) with non-empty meshes;
        rgba is either a single color or an (nvert, 4) array."""
        verts, norms, colors, faces = [], [], [], []
        off = 0
        for mesh, rgba in parts:
            verts.append(mesh.vertices)
            norms.append(mesh.normals)
            rgba = np.asarray(rgba, np.float32)
            if rgba.ndim == 1:
                rgba = np.tile(rgba, (len(mesh.vertices), 1))
            colors.append(rgba)
            faces.append(mesh.faces + off)
            off += len(mesh.vertices)
        v = np.vstack(verts)
        self.faces = np.vstack(faces).astype(np.uint32)
        self.centroids = v[self.faces].mean(axis=1)  # (F, 3)

        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        data = np.hstack([v, np.vstack(norms),
                          np.vstack(colors)]).astype(np.float32)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data,
                        GL.GL_STATIC_DRAW)
        stride = 40  # 3 pos + 3 norm + 4 color floats
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, stride,
                                 ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, stride,
                                 ctypes.c_void_p(12))
        GL.glEnableVertexAttribArray(2)
        GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, False, stride,
                                 ctypes.c_void_p(24))
        self._ebo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.faces.nbytes,
                        None, GL.GL_STREAM_DRAW)
        GL.glBindVertexArray(0)

    def draw(self, view_matrix):
        if not len(self.faces):
            return
        # view-space z of each face centroid; most negative = farthest
        vz = self.centroids @ view_matrix[2, :3]
        order = np.argsort(vz)  # far to near
        idx = np.ascontiguousarray(self.faces[order])
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx,
                        GL.GL_STREAM_DRAW)
        GL.glDrawElements(GL.GL_TRIANGLES, idx.size,
                          GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

    def delete(self):
        GL.glDeleteBuffers(2, [self._vbo, self._ebo])
        GL.glDeleteVertexArrays(1, [self.vao])
        self.faces = np.zeros((0, 3), np.uint32)


class OffscreenTarget:
    """MSAA FBO + resolve FBO for thumbnails and export."""

    def __init__(self, width, height, samples=4):
        self.width, self.height = width, height
        maxs = GL.glGetIntegerv(GL.GL_MAX_SAMPLES)
        self.samples = max(0, min(samples, int(maxs)))
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        self._rb_c = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self._rb_c)
        GL.glRenderbufferStorageMultisample(
            GL.GL_RENDERBUFFER, self.samples, GL.GL_RGBA8, width, height)
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
            GL.GL_RENDERBUFFER, self._rb_c)
        self._rb_d = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self._rb_d)
        GL.glRenderbufferStorageMultisample(
            GL.GL_RENDERBUFFER, self.samples, GL.GL_DEPTH_COMPONENT24,
            width, height)
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
            GL.GL_RENDERBUFFER, self._rb_d)
        self.resolve_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.resolve_fbo)
        self._rb_r = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self._rb_r)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8,
                                 width, height)
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
            GL.GL_RENDERBUFFER, self._rb_r)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Incomplete framebuffer (status {status})")

    def bind(self):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, self.width, self.height)

    def read_pixels(self):
        """Resolve MSAA and return an (h, w, 4) uint8 array (top row first)."""
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.resolve_fbo)
        GL.glBlitFramebuffer(0, 0, self.width, self.height,
                             0, 0, self.width, self.height,
                             GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.resolve_fbo)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        raw = GL.glReadPixels(0, 0, self.width, self.height,
                              GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        img = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.height, self.width, 4)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        return img[::-1].copy()

    def blit_to(self, target_fbo, width, height):
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, target_fbo)
        GL.glBlitFramebuffer(0, 0, self.width, self.height,
                             0, 0, width, height,
                             GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)

    def delete(self):
        GL.glDeleteRenderbuffers(3, [self._rb_c, self._rb_d, self._rb_r])
        GL.glDeleteFramebuffers(2, [self.fbo, self.resolve_fbo])


class Renderer:
    """Owns all GL objects; must be used with its GL context current."""

    def __init__(self):
        self.prog_inst = _compile(_VERT_INSTANCED, _FRAG_PHONG)
        self.prog_surf = _compile(_VERT_SURFACE, _FRAG_PHONG)
        self.prog_surfc = _compile(_VERT_SURFACE_VC, _FRAG_PHONG)
        self.prog_halo = _compile(_VERT_INSTANCED, _FRAG_HALO)
        self.prog_line = _compile(_VERT_LINES, _FRAG_LINES)
        self.prog_label = _compile(_VERT_LABEL, _FRAG_LABEL)
        self.sphere = _Primitive(*meshes.uv_sphere())
        self.cylinder = _Primitive(*meshes.cylinder())
        self.cone = _Primitive(*meshes.cone())
        self.disc = _Primitive(*meshes.disc())
        self._line_vao = GL.glGenVertexArrays(1)
        self._line_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._line_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 24,
                                 ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 24,
                                 ctypes.c_void_p(12))
        GL.glBindVertexArray(0)
        # atom-label resources (texture atlas + billboarded quads)
        self._label_vao = GL.glGenVertexArrays(1)
        self._label_vbo = GL.glGenBuffers(1)
        self._label_ebo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._label_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._label_vbo)
        stride = 8 * 4  # anchor 3, corner 2, uv 2, zoff 1
        for loc, size, off in ((0, 3, 0), (1, 2, 12), (2, 2, 20),
                               (3, 1, 28)):
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False,
                                     stride, ctypes.c_void_p(off))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._label_ebo)
        GL.glBindVertexArray(0)
        self._label_tex = None
        self._label_nidx = 0
        self._labels_key = None
        # GPU meshes for surface-bearing items, keyed by (item_id, tag)
        self._surfaces = {}
        # depth-sorted transparent meshes, keyed by item_id
        self._transparent = {}

    # ----------------------------------------------------- surface caching

    def surface_gpu(self, key, mesh):
        """Get/create GPU buffers for a mesh; re-upload when replaced."""
        entry = self._surfaces.get(key)
        if entry is not None and entry[1] is mesh:
            return entry[0]
        if entry is not None:
            entry[0].delete()
        gpu = SurfaceGPU(mesh)
        self._surfaces[key] = (gpu, mesh)
        return gpu

    def colored_surface_gpu(self, key, mesh, colors):
        """Get/create GPU buffers for a per-vertex-colored mesh.  The
        colors are tied to the mesh (both are rebuilt together), so mesh
        identity is the cache-validity key."""
        entry = self._surfaces.get(key)
        if entry is not None and entry[1] is mesh:
            return entry[0]
        if entry is not None:
            entry[0].delete()
        gpu = ColoredSurfaceGPU(mesh, colors)
        self._surfaces[key] = (gpu, mesh)
        return gpu

    def _transparent_gpu(self, item_id, key, make_parts):
        """Get/create the sorted transparent mesh for an item.  `key`
        is the cache-validity key; `make_parts` builds the (mesh, rgba)
        list only when the cache is stale."""
        entry = self._transparent.get(item_id)
        if entry is not None and entry[1] == key:
            return entry[0]
        if entry is not None:
            entry[0].delete()
            del self._transparent[item_id]
        gpu = TransparentSurface(make_parts())
        self._transparent[item_id] = (gpu, key)
        return gpu

    def drop_surfaces(self, item_id):
        for key in [k for k in self._surfaces if k[0] == item_id]:
            self._surfaces.pop(key)[0].delete()
        entry = self._transparent.pop(item_id, None)
        if entry is not None:
            entry[0].delete()

    def drop_all_surfaces(self):
        for key in list(self._surfaces):
            self._surfaces.pop(key)[0].delete()
        for entry in self._transparent.values():
            entry[0].delete()
        self._transparent.clear()

    # -------------------------------------------------------------- drawing

    def _set_matrices(self, prog, camera, aspect):
        GL.glUseProgram(prog)
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(prog, "u_view"), 1, GL.GL_TRUE,
            camera.view_matrix())
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(prog, "u_proj"), 1, GL.GL_TRUE,
            camera.proj_matrix(aspect))

    # ----------------------------------------------------------- atom labels

    def set_atom_labels(self, labels, world_height=0.95):
        """labels: list of (position(3,), text, z_offset).  Rebuilds the
        GPU data only when the label set actually changed."""
        key = (tuple(t for _p, t, _z in labels),
               np.array([p for p, _t, _z in labels]).tobytes() if labels
               else b"",
               tuple(round(z, 3) for _p, _t, z in labels))
        if key == self._labels_key:
            return
        self._labels_key = key
        if not labels:
            self._label_nidx = 0
            return

        import cairo
        font_px = 44
        pad = 8
        unique = sorted({t for _p, t, _z in labels})
        meas = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1))
        meas.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL,
                              cairo.FONT_WEIGHT_BOLD)
        meas.set_font_size(font_px)
        atlas_w = 1024
        x = y = row_h = 0
        slots = {}
        for t in unique:
            ext = meas.text_extents(t)
            w = int(np.ceil(ext.width)) + 2 * pad
            h = int(np.ceil(ext.height)) + 2 * pad
            if x + w > atlas_w:
                x, y, row_h = 0, y + row_h, 0
            slots[t] = (x, y, w, h, ext.x_bearing, ext.y_bearing)
            x += w
            row_h = max(row_h, h)
        atlas_h = y + row_h

        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, atlas_w, atlas_h)
        cr = cairo.Context(surf)
        cr.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL,
                            cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(font_px)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        for t, (sx, sy, _w, _h, bx, by) in slots.items():
            cr.new_path()
            cr.move_to(sx + pad - bx, sy + pad - by)
            cr.text_path(t)
            cr.set_source_rgba(1.0, 1.0, 1.0, 1.0)  # white outline
            cr.set_line_width(5.0)
            cr.stroke_preserve()
            cr.set_source_rgba(0.05, 0.05, 0.05, 1.0)  # dark fill
            cr.fill()
        surf.flush()
        buf = np.ndarray((atlas_h, surf.get_stride() // 4, 4),
                         dtype=np.uint8,
                         buffer=surf.get_data())[:, :atlas_w]
        rgba = np.ascontiguousarray(buf[..., [2, 1, 0, 3]])  # BGRA->RGBA

        if self._label_tex is None:
            self._label_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._label_tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER,
                           GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, atlas_w,
                        atlas_h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba)

        scale = world_height / font_px  # world units per atlas pixel
        verts = np.empty((len(labels) * 4, 8), np.float32)
        idx = np.empty((len(labels), 6), np.uint32)
        for n, (pos, text, zoff) in enumerate(labels):
            sx, sy, w, h, _bx, _by = slots[text]
            w2 = 0.5 * w * scale
            h2 = 0.5 * h * scale
            u0, u1 = sx / atlas_w, (sx + w) / atlas_w
            v0, v1 = sy / atlas_h, (sy + h) / atlas_h
            base = n * 4
            for k, (cx, cy, u, v) in enumerate(((-w2, h2, u0, v0),
                                                (w2, h2, u1, v0),
                                                (w2, -h2, u1, v1),
                                                (-w2, -h2, u0, v1))):
                verts[base + k] = (*pos, cx, cy, u, v, zoff)
            idx[n] = (base, base + 2, base + 1, base, base + 3, base + 2)
        GL.glBindVertexArray(self._label_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._label_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts,
                        GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._label_ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx,
                        GL.GL_DYNAMIC_DRAW)
        GL.glBindVertexArray(0)
        self._label_nidx = idx.size

    def _draw_labels(self, camera, aspect):
        if not self._label_nidx:
            return
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
        self._set_matrices(self.prog_label, camera, aspect)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._label_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self.prog_label, "u_tex"),
                       0)
        GL.glBindVertexArray(self._label_vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self._label_nidx,
                          GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
                               GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)

    def _draw_axes(self, camera, aspect, length):
        """Cartesian axes at the world origin with billboarded labels."""
        r = max(length * 0.008, 0.045)
        cyls, cones = [], []
        for d, color, _ in _AXES:
            shaft = d * length * 0.88
            m = meshes.bond_matrix((0.0, 0.0, 0.0), shaft, r)
            cyls.append(np.concatenate([
                m.T.flatten(), [*color, 1.0],
                [r, r, length * 0.88, 0.0]]))
            mc = meshes.bond_matrix(shaft, d * length, r * 2.4)
            cones.append(np.concatenate([
                mc.T.flatten(), [*color, 1.0],
                [r * 2.4, r * 2.4, length * 0.12, 0.0]]))
        self._set_matrices(self.prog_inst, camera, aspect)
        GL.glUniform1f(GL.glGetUniformLocation(self.prog_inst,
                                               "u_twosided"), 0.0)
        self.cylinder.draw(np.array(cyls, np.float32))
        self.cone.draw(np.array(cones, np.float32))

        # letter labels just past the tips, always facing the camera
        to_world = quat_to_mat3(camera.rotation).T
        s = length * 0.026
        lines = []
        for d, color, glyph in _AXES:
            pos = d * (length + 4.0 * s)
            for u0, v0, u1, v1 in _GLYPHS[glyph]:
                p0 = pos + to_world @ (u0 * s, v0 * s, 0.0)
                p1 = pos + to_world @ (u1 * s, v1 * s, 0.0)
                lines.append([*p0, *color])
                lines.append([*p1, *color])
        seg = np.array(lines, np.float32)
        self._set_matrices(self.prog_line, camera, aspect)
        GL.glBindVertexArray(self._line_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, seg.nbytes, seg,
                        GL.GL_STREAM_DRAW)
        GL.glLineWidth(3.5)  # bold labels
        GL.glDrawArrays(GL.GL_LINES, 0, len(seg))
        GL.glLineWidth(1.0)
        GL.glBindVertexArray(0)

    def render(self, scene, item, camera, width, height,
               background=(1.0, 1.0, 1.0), transparent=False, axes=None,
               labels=False):
        """Draw structure + one item into the currently bound framebuffer.

        axes: None, or the axis length (bohr) to draw Cartesian axes at
        the world origin (used by the main view only).
        labels: draw the atom labels set via set_atom_labels (main view
        only; thumbnails and exports never pass this)."""
        aspect = width / max(height, 1)
        if transparent:
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        else:
            GL.glClearColor(*background, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_MULTISAMPLE)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_CULL_FACE)

        spheres, cyls, lines, halos = scene.structure_arrays()

        self._set_matrices(self.prog_inst, camera, aspect)
        GL.glUniform1f(GL.glGetUniformLocation(self.prog_inst,
                                               "u_twosided"), 0.0)
        self.sphere.draw(spheres)
        self.cylinder.draw(cyls)

        if len(lines):
            self._set_matrices(self.prog_line, camera, aspect)
            GL.glBindVertexArray(self._line_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, lines.nbytes, lines,
                            GL.GL_STREAM_DRAW)
            GL.glDrawArrays(GL.GL_LINES, 0, len(lines))
            GL.glBindVertexArray(0)

        if axes is not None:
            self._draw_axes(camera, aspect, axes)
        if labels:
            self._draw_labels(camera, aspect)

        if len(halos):
            # translucent rim-glow shells marking the selected atoms
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
                               GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDepthMask(GL.GL_FALSE)
            self._set_matrices(self.prog_halo, camera, aspect)
            self.sphere.draw(halos)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)

        if item is not None:
            self._draw_item(scene, item, camera, aspect)
        GL.glUseProgram(0)

    def _draw_item(self, scene, item, camera, aspect):
        settings = scene.settings
        if item.kind in ("orbital", "density", "spin"):
            pre = "iso" if item.kind == "orbital" else "dens"
            alpha = settings[pre + "_alpha"]
            colors = (settings[pre + "_color_pos"],
                      settings[pre + "_color_neg"])
            parts = [(mesh, [*color, alpha])
                     for mesh, color in ((item.mesh_pos, colors[0]),
                                         (item.mesh_neg, colors[1]))
                     if mesh is not None and not mesh.empty]
            if not parts:
                return
            if alpha < 0.999:
                # both lobes merged and depth-sorted for correct blending
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
                               GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glDepthMask(GL.GL_FALSE)
                GL.glDisable(GL.GL_CULL_FACE)
                self._set_matrices(self.prog_surfc, camera, aspect)
                GL.glUniform1f(GL.glGetUniformLocation(self.prog_surfc,
                                                       "u_twosided"), 1.0)
                key = tuple((id(mesh), tuple(rgba))
                            for mesh, rgba in parts)
                self._transparent_gpu(item.id, key, lambda: parts).draw(
                    camera.view_matrix())
                GL.glDepthMask(GL.GL_TRUE)
                GL.glDisable(GL.GL_BLEND)
                GL.glEnable(GL.GL_CULL_FACE)
                return
            GL.glDisable(GL.GL_CULL_FACE)
            self._set_matrices(self.prog_surf, camera, aspect)
            GL.glUniform1f(GL.glGetUniformLocation(self.prog_surf,
                                                   "u_twosided"), 1.0)
            uc = GL.glGetUniformLocation(self.prog_surf, "u_color")
            for mesh, color, tag in ((item.mesh_pos, colors[0], "pos"),
                                     (item.mesh_neg, colors[1], "neg")):
                if mesh is None or mesh.empty:
                    continue
                GL.glUniform4f(uc, *color, alpha)
                self.surface_gpu((item.id, tag), mesh).draw()
            GL.glEnable(GL.GL_CULL_FACE)
        elif item.kind == "esp":
            mesh, colors = item.mesh_pos, item.esp_colors
            if mesh is None or mesh.empty or colors is None:
                return
            alpha = settings["esp_alpha"]
            if alpha < 0.999:
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
                               GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glDepthMask(GL.GL_FALSE)
                GL.glDisable(GL.GL_CULL_FACE)
                self._set_matrices(self.prog_surfc, camera, aspect)
                GL.glUniform1f(GL.glGetUniformLocation(self.prog_surfc,
                                                       "u_twosided"), 1.0)
                key = (id(mesh), id(colors), round(alpha, 3))

                def make():
                    rgba = colors.copy()
                    rgba[:, 3] = alpha
                    return [(mesh, rgba)]

                self._transparent_gpu(item.id, key, make).draw(
                    camera.view_matrix())
                GL.glDepthMask(GL.GL_TRUE)
                GL.glDisable(GL.GL_BLEND)
                GL.glEnable(GL.GL_CULL_FACE)
                return
            GL.glDisable(GL.GL_CULL_FACE)
            self._set_matrices(self.prog_surfc, camera, aspect)
            GL.glUniform1f(GL.glGetUniformLocation(self.prog_surfc,
                                                   "u_twosided"), 1.0)
            self.colored_surface_gpu((item.id, "esp"), mesh,
                                     colors).draw()
            GL.glEnable(GL.GL_CULL_FACE)
        elif item.kind == "vector":
            cyls, cones = scene.item_instances(item)
            self._set_matrices(self.prog_inst, camera, aspect)
            GL.glUniform1f(GL.glGetUniformLocation(self.prog_inst,
                                                   "u_twosided"), 0.0)
            if cyls:
                self.cylinder.draw(np.array(cyls, np.float32))
            if cones:
                self.cone.draw(np.array(cones, np.float32))
        elif item.kind == "plane":
            discs = scene.item_instances(item)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
                               GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDepthMask(GL.GL_FALSE)
            GL.glDisable(GL.GL_CULL_FACE)
            self._set_matrices(self.prog_inst, camera, aspect)
            GL.glUniform1f(GL.glGetUniformLocation(self.prog_inst,
                                                   "u_twosided"), 1.0)
            self.disc.draw(np.array(discs, np.float32))
            GL.glEnable(GL.GL_CULL_FACE)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)
        elif item.kind == "polyhedron":
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
                               GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDepthMask(GL.GL_FALSE)
            self._set_matrices(self.prog_surf, camera, aspect)
            GL.glUniform1f(GL.glGetUniformLocation(self.prog_surf,
                                                   "u_twosided"), 1.0)
            GL.glUniform4f(GL.glGetUniformLocation(self.prog_surf,
                                                   "u_color"),
                           *item.color, item.alpha)
            gpu = self.surface_gpu((item.id, "hull"), item.mesh)
            # convex shape: far side first, then near side, so the two
            # layers blend in the correct order
            GL.glEnable(GL.GL_CULL_FACE)
            GL.glCullFace(GL.GL_FRONT)
            gpu.draw()
            GL.glCullFace(GL.GL_BACK)
            gpu.draw()
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)
            # edges
            edges = item.edges
            if len(edges):
                self._set_matrices(self.prog_line, camera, aspect)
                col = np.array(item.color, np.float32) * 0.5
                seg = np.empty((edges.shape[0] * 2, 6), np.float32)
                seg[:, :3] = edges.reshape(-1, 3)
                seg[:, 3:] = col
                GL.glBindVertexArray(self._line_vao)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, seg.nbytes, seg,
                                GL.GL_STREAM_DRAW)
                GL.glDrawArrays(GL.GL_LINES, 0, len(seg))
                GL.glBindVertexArray(0)
