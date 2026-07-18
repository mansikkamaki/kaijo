# SPDX-License-Identifier: GPL-3.0-or-later
"""Main 3D view: a GtkGLArea with trackball interaction.

Mouse bindings (per the design spec):
    drag                rotate (virtual trackball)
    scroll              zoom
    shift + drag        zoom (vertical mouse motion)
    ctrl  + drag        pan (drag molecule on screen)
    alt   + drag        rotate within the screen plane
    click (no drag)     select/deselect atom
"""

import numpy as np

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, Gtk
from OpenGL import GL

from ..core import elements
from ..core.elements import BOHR
from ..render.renderer import Renderer, OffscreenTarget


class GLView(Gtk.GLArea):
    def __init__(self, scene, camera):
        super().__init__()
        self.scene = scene
        self.camera = camera
        self.renderer = None
        self.current_item = None       # item shown alongside the structure
        self.show_axes = False         # Cartesian axes, main view only
        self.show_symbols = False      # element-symbol labels
        self.show_indices = False      # atom-index labels
        self.on_atom_clicked = None    # callback(atom_index or None)
        self.on_interaction_end = None
        self.gl_error = None
        self._msaa = None
        self._thumb_target = None      # reused offscreen buffer for thumbnails
        self._labels = []              # cached label build (see _on_render)
        self._labels_dirty = True
        self._drag = None              # (x, y, mode)
        self._moved = False

        self.set_required_version(3, 3)
        self.set_has_depth_buffer(False)   # we render via our own FBO
        self.set_auto_render(True)
        self.set_can_focus(True)
        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK
                        | Gdk.EventMask.BUTTON_RELEASE_MASK
                        | Gdk.EventMask.POINTER_MOTION_MASK
                        | Gdk.EventMask.SCROLL_MASK
                        | Gdk.EventMask.SMOOTH_SCROLL_MASK)
        self.connect("realize", self._on_realize)
        self.connect("unrealize", self._on_unrealize)
        self.connect("render", self._on_render)
        self.connect("button-press-event", self._on_press)
        self.connect("button-release-event", self._on_release)
        self.connect("motion-notify-event", self._on_motion)
        self.connect("scroll-event", self._on_scroll)

    # ------------------------------------------------------------------- GL

    def _on_realize(self, area):
        area.make_current()
        if area.get_error() is not None:
            self.gl_error = str(area.get_error())
            return
        try:
            self.renderer = Renderer()
        except Exception as exc:  # noqa: BLE001
            self.gl_error = str(exc)

    def _on_unrealize(self, area):
        area.make_current()
        if self.renderer is not None:
            self.renderer.drop_all_surfaces()
        self.renderer = None
        self._msaa = None
        self._thumb_target = None

    def _viewport_size(self):
        scale = self.get_scale_factor()
        alloc = self.get_allocation()
        return max(alloc.width * scale, 1), max(alloc.height * scale, 1)

    def _on_render(self, area, context):
        if self.renderer is None:
            return True
        w, h = self._viewport_size()
        target_fbo = GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING)
        if self._msaa is None or self._msaa.width != w \
                or self._msaa.height != h:
            if self._msaa is not None:
                self._msaa.delete()
            self._msaa = OffscreenTarget(w, h)
        draw_labels = self.show_symbols or self.show_indices
        if draw_labels:
            # labels are billboarded in the shader, so they are independent
            # of the camera; rebuild only when the molecule/settings change
            if self._labels_dirty:
                self._labels = self._build_labels()
                self._labels_dirty = False
            self.renderer.set_atom_labels(self._labels)
        self._msaa.bind()
        self.renderer.render(self.scene, self.current_item, self.camera,
                             w, h,
                             background=self.scene.settings["background"],
                             axes=self.axes_length() if self.show_axes
                             else None,
                             labels=draw_labels)
        self._msaa.blit_to(target_fbo, w, h)
        GL.glViewport(0, 0, w, h)
        return True

    def _build_labels(self):
        """(position, text, z_offset) per labelled atom; follows the
        hydrogen-visibility setting so toggling it updates the labels."""
        mol = self.scene.molecule
        if mol is None:
            return []
        s = self.scene.settings
        show_h = s["show_hydrogens"]
        rep = s["representation"]
        bond_r = s["bond_radius"] / BOHR
        out = []
        for i in range(mol.natoms):
            if not mol.visible[i]:
                continue
            z = mol.numbers[i]
            if z == 1 and not show_h:
                continue
            sym = elements.z_to_symbol(z)
            if self.show_symbols and self.show_indices:
                text = f"{sym}({i + 1})"
            elif self.show_symbols:
                text = sym
            else:
                text = str(i + 1)
            if rep == "ball-and-stick":
                r = self.scene.atom_radius(z)
            elif rep == "sticks":
                r = bond_r * 1.7
            else:
                r = 0.0
            out.append((mol.coords[i], text, r + 0.35))
        return out

    def invalidate_labels(self):
        """Force the atom labels to be rebuilt on the next render."""
        self._labels_dirty = True

    def axes_length(self):
        mol = self.scene.molecule
        if mol is None:
            return 9.0
        return max(mol.extent() * 0.825, 6.0)

    def render_offscreen(self, item, width, height, background,
                         transparent, camera=None, reuse=False):
        """Render into an offscreen buffer; returns (h, w, 4) uint8 RGBA.

        With ``reuse=True`` a single offscreen target is kept and re-bound
        across calls (used for the same-sized preview thumbnails, which are
        rendered many times); otherwise a throwaway target is allocated
        (used by image export, whose sizes vary and which runs once)."""
        self.make_current()
        if self.renderer is None:
            return None
        if reuse:
            tgt = self._thumb_target
            if tgt is None or tgt.width != width or tgt.height != height:
                if tgt is not None:
                    tgt.delete()
                tgt = self._thumb_target = OffscreenTarget(width, height)
        else:
            tgt = OffscreenTarget(width, height)
        try:
            tgt.bind()
            self.renderer.render(self.scene, item, camera or self.camera,
                                 width, height, background=background,
                                 transparent=transparent)
            return tgt.read_pixels()
        finally:
            if not reuse:
                tgt.delete()
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def drop_item_surfaces(self, item_id):
        if self.renderer is not None and self.get_realized():
            self.make_current()
            self.renderer.drop_surfaces(item_id)

    # ------------------------------------------------------------ interaction

    def _on_press(self, area, event):
        self.grab_focus()
        if event.button == 1:
            state = event.state
            if state & Gdk.ModifierType.CONTROL_MASK:
                mode = "pan"
            elif state & Gdk.ModifierType.SHIFT_MASK:
                mode = "zoom"
            elif state & Gdk.ModifierType.MOD1_MASK:
                mode = "roll"
            else:
                mode = "rotate"
            self._drag = (event.x, event.y, mode)
            self._moved = False
        return True

    def _on_release(self, area, event):
        if event.button == 1 and self._drag is not None:
            if not self._moved and self.on_atom_clicked:
                atom = self._pick(event.x, event.y)
                self.on_atom_clicked(atom)
            elif self._moved and self.on_interaction_end:
                self.on_interaction_end()
            self._drag = None
        return True

    def _on_motion(self, area, event):
        if self._drag is None:
            return False
        x0, y0, mode = self._drag
        dx, dy = event.x - x0, event.y - y0
        if not self._moved and dx * dx + dy * dy < 9:
            return True
        self._moved = True
        alloc = self.get_allocation()
        w, h = max(alloc.width, 1), max(alloc.height, 1)
        if mode == "rotate":
            s = min(w, h)
            nx0 = (2 * x0 - w) / s
            ny0 = (h - 2 * y0) / s
            nx1 = (2 * event.x - w) / s
            ny1 = (h - 2 * event.y) / s
            self.camera.trackball(nx0, ny0, nx1, ny1)
        elif mode == "pan":
            self.camera.pan(-dx, -dy, h)
        elif mode == "zoom":
            self.camera.zoom(np.exp(dy * 0.005))
        elif mode == "roll":
            cx, cy = w / 2, h / 2
            a0 = np.arctan2(y0 - cy, x0 - cx)
            a1 = np.arctan2(event.y - cy, event.x - cx)
            self.camera.roll(a0 - a1)
        self._drag = (event.x, event.y, mode)
        return True

    def _on_scroll(self, area, event):
        delta = 0.0
        if event.direction == Gdk.ScrollDirection.UP:
            delta = -1.0
        elif event.direction == Gdk.ScrollDirection.DOWN:
            delta = 1.0
        elif event.direction == Gdk.ScrollDirection.SMOOTH:
            delta = event.delta_y
        if delta:
            self.camera.zoom(np.exp(delta * 0.12))
            if self.on_interaction_end:
                self.on_interaction_end()
        return True

    def _pick(self, x, y):
        alloc = self.get_allocation()
        origin, direction = self.camera.pixel_ray(
            x, y, max(alloc.width, 1), max(alloc.height, 1))
        return self.scene.pick_atom(origin, direction)
