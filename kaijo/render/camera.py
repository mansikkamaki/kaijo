# SPDX-License-Identifier: GPL-3.0-or-later
"""Orthographic trackball camera.

Orthographic projection is used throughout: it is the norm for
publication-quality molecular graphics and makes exported image sizes an
exact multiple of the preview size.
"""

import numpy as np


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), *(axis * s)])


def quat_to_mat3(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_mat3(m):
    """Quaternion for a rotation matrix (Shepperd's method); inverse of
    quat_to_mat3."""
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s,
                      (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] >= m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s,
                      (m[0, 1] + m[1, 0]) / s, 0.25 * s,
                      (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s,
                      (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def _trackball_point(x, y):
    """Map normalised screen coords (-1..1) onto a virtual sphere."""
    d2 = x * x + y * y
    r2 = 0.64  # sphere radius^2 (0.8^2)
    if d2 < r2 / 2:
        z = np.sqrt(r2 - d2)
    else:
        z = r2 / 2 / np.sqrt(d2)
    return np.array([x, y, z])


class Camera:
    """Shared by the main view and every preview cell (single orientation)."""

    def __init__(self):
        self.rotation = np.array([1.0, 0.0, 0.0, 0.0])  # world->view quat
        self.center = np.zeros(3)      # look-at point, world coords (bohr)
        self.half_height = 10.0        # ortho half-height, bohr
        self.depth = 100.0             # ortho z half-range
        self.on_change = None

    def fit(self, center, extent):
        self.center = np.asarray(center, dtype=np.float64).copy()
        self.half_height = max(extent * 1.15, 1.0)
        self.depth = max(extent * 4.0, 10.0)
        self._changed()

    def _changed(self):
        if self.on_change:
            self.on_change()

    # ------------------------------------------------------------- matrices

    def view_matrix(self):
        m = np.eye(4)
        rot = quat_to_mat3(self.rotation)
        m[:3, :3] = rot
        m[:3, 3] = -rot @ self.center
        return m.astype(np.float32)

    def proj_matrix(self, aspect):
        h = self.half_height
        w = h * aspect
        d = self.depth
        m = np.diag([1.0 / w, 1.0 / h, -1.0 / d, 1.0]).astype(np.float32)
        return m

    # ---------------------------------------------------------- interaction

    def trackball(self, x0, y0, x1, y1):
        """Rotate; coords normalised to -1..1 (y up)."""
        if x0 == x1 and y0 == y1:
            return
        p0 = _trackball_point(x0, y0)
        p1 = _trackball_point(x1, y1)
        axis = np.cross(p0, p1)
        dot = min(max(np.dot(p0, p1) /
                      (np.linalg.norm(p0) * np.linalg.norm(p1)), -1), 1)
        angle = np.arccos(dot) * 1.6
        # axis is in view space; convert to world space for pre-multiply
        q = quat_from_axis_angle(axis, angle)
        self.rotation = quat_mul(q, self.rotation)
        self.rotation /= np.linalg.norm(self.rotation)
        self._changed()

    def set_view_direction(self, forward, up=None):
        """Look along `forward` (world coords); e.g. (1,0,0) views the
        molecule along the positive x axis."""
        f = np.asarray(forward, dtype=np.float64)
        f = f / np.linalg.norm(f)
        if up is None:
            up = (0.0, 1.0, 0.0) if abs(f[2]) > 0.9 else (0.0, 0.0, 1.0)
        u0 = np.asarray(up, dtype=np.float64)
        right = np.cross(f, u0)
        right /= np.linalg.norm(right)
        u = np.cross(right, f)
        rot = np.array([right, u, -f])  # rows: view x, y, z in world coords
        self.rotation = quat_from_mat3(rot)
        self._changed()

    def roll(self, angle):
        """Rotate within the screen plane (view z axis)."""
        q = quat_from_axis_angle([0.0, 0.0, 1.0], angle)
        self.rotation = quat_mul(q, self.rotation)
        self.rotation /= np.linalg.norm(self.rotation)
        self._changed()

    def pan(self, dx, dy, viewport_h):
        """Drag the molecule; dx, dy in pixels."""
        scale = 2.0 * self.half_height / max(viewport_h, 1)
        rot = quat_to_mat3(self.rotation)
        shift = rot.T @ np.array([-dx * scale, dy * scale, 0.0])
        self.center += shift
        self._changed()

    def zoom(self, factor):
        self.half_height = min(max(self.half_height * factor, 0.5), 2000.0)
        self._changed()

    # ------------------------------------------------------------- picking

    def pixel_ray(self, px, py, width, height):
        """Ray (origin, direction) in world coords through pixel (px, py)."""
        aspect = width / max(height, 1)
        nx = (2.0 * px / width - 1.0) * self.half_height * aspect
        ny = (1.0 - 2.0 * py / height) * self.half_height
        rot = quat_to_mat3(self.rotation)
        origin = self.center + rot.T @ np.array([nx, ny, self.depth])
        direction = rot.T @ np.array([0.0, 0.0, -1.0])
        return origin, direction
