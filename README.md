# Kaijo

**Version 0.3.0**

A fast, simple molecular orbital and molecular structure visualization
program.

**Developer:** Akseli Mansikkamäki.  Kaijo was constructed with AI using
Claude Code (models: Claude Opus 4.8 and Claude Fable 5.0).

**License:** GNU General Public License, version 3.0 or later
(GPL-3.0-or-later); see the [LICENSE](LICENSE) file for the full text.
Copyright © 2026 Akseli Mansikkamäki.

## Requirements

* Python 3.8+
* GTK 3 with GObject introspection (`python3-gi`, `gir1.2-gtk-3.0`)
* OpenGL 3.3 capable driver
* Python packages: `numpy`, `scipy`, `PyOpenGL`, `scikit-image`, `Pillow`

```sh
pip install --user numpy scipy PyOpenGL scikit-image Pillow
```

## Running

```sh
./kaijo-run [file.molden | file.fchk | file.xyz]
```

or `python3 -m kaijo.main [file]` from the project directory.

## Supported files

* **molden** — geometry, GTO basis and molecular orbitals.  Spherical
  bases up to g functions ([5D]/[7F]/[9G]).  ORCA's documented deviations
  from the molden standard (contraction-coefficient normalisation, sign
  of the f±3/g±3/g±4 components) are detected and corrected automatically.
  Gaussian-derived files that write an ECP-reduced charge in the atomic
  number column are recognised by their element symbols.
* **fchk** — Gaussian formatted checkpoint: geometry, GTO basis and
  restricted/unrestricted molecular orbitals, including ECP calculations
  (the true atomic number is taken from `Atomic numbers`).  Spherical
  bases up to g functions; combined SP shells are handled.  Cartesian
  d/f/g shells are reported as unsupported.  The (potentially very large)
  coefficient blocks are parsed with numpy for molden-class speed.
* **xyz** — geometry only; both the standard form (count + comment line)
  and a bare element/coordinate listing are accepted.

New formats can be added in `kaijo/formats/` by subclassing
`FormatHandler`; new grid types in `kaijo/core/grid.py` via `GRID_TYPES`.

## Using the program

**Layout.**  Left: the options bar with the orbital list (index, energy in
hartree, occupation, spin), the *Visible items* and *Camera angle*
sections, the selection/geometry tools and the object properties panel.
Centre: the main 3D view.  Right: the preview grid (as many columns as
fit its width — drag the divider to resize; the grid scrolls).  Bottom:
status line with the progress bar and the scratch-disk usage.  Each
preview cell has an [x] button that removes it from the grid (a closed
orbital is also deselected in the orbital list; the removal is undoable).

**Visible items.**  *Hydrogen atoms* (on by default) shows/hides the
hydrogens and their bonds in the main view and every preview cell.
*Cartesian axes* (off by default) draws the x/y/z axes as thin labelled
vectors centred on the (0, 0, 0) coordinate origin — in the main view
only, never in the preview grid or exported images.  *Element symbols*
and *Atom indices* (both off by default) label each atom in the main
view: "C" with symbols only, "6" (the atom's number in the coordinate
listing) with indices only, "C(6)" with both.  Hidden hydrogens are
never labelled, and the labels follow the hydrogen-visibility toggle.
*Draw double/triple bonds* (on by default) renders bond orders as
separate lines in all views; when off, every bond is drawn as a single
line.

**Camera angle.**  The x/y/z buttons view the molecule along the
corresponding axis in the positive direction; x*/y*/z* along the
negative direction.  Zoom, pan and the preview grid follow as usual.

**Mouse (main view).**

| action              | effect                        |
|---------------------|-------------------------------|
| drag                | rotate (trackball)            |
| scroll              | zoom                          |
| shift + drag        | zoom                          |
| ctrl + drag         | pan                           |
| alt + drag          | rotate in the screen plane    |
| click on an atom    | select / deselect it          |

There is a single molecular orientation: rotating the main view rotates
every preview cell as well.

**Orbitals.**  Select rows in the orbital list (Ctrl/Shift-click, or type
a range such as `150-160`, `42a`, `12,15,20-25` into the entry) and press
*Visualize selected orbitals*.  You are prompted for the grid spacing,
margin and isovalue (sensible defaults are remembered).  Each orbital
appears as a preview cell as soon as its surface is ready; clicking a cell
shows it in the main view.  Pressing *Visualize* again with a changed
selection adds the new orbitals and removes the deselected ones; already
computed volumes are cached on scratch disk, so changing only the isovalue
re-extracts the surfaces almost instantly.

**Calculate (density / spin density / ESP).**  The green buttons in the
*Calculate* section compute scalar fields from the loaded orbitals and
add them to the preview grid like orbitals:

* *Density* — the electron density Σ nᵢ|ψᵢ|², shown as an isosurface
  (default isovalue 0.002 e/bohr³).
* *Spin density* — the difference between the alpha and beta densities;
  needs an unrestricted file (the button is disabled otherwise).  The
  positive and negative parts use the density colors.
* *ESP* — the electrostatic potential mapped onto an electron-density
  isosurface with a fixed red-white-blue gradient (red = negative,
  white = zero, blue = positive potential; the range adapts to the
  surface and is shown in the preview label).  The electronic part is
  solved from the gridded density by an FFT Poisson convolution; for
  files from ECP calculations the effective nuclear charges are
  inferred automatically so that nuclei and electrons balance.

Each button prompts for its grid and isovalue.  The densities and the
ESP each keep their own grid and surface settings, separate from the
orbital ones; selecting a density (or ESP) cell shows those settings in
the *Object / surface properties* panel.  The ESP colors are fixed and
cannot be edited.  All field volumes are cached on scratch disk, so
isovalue changes only re-run the surface extraction.

**Geometry tools.**  Click atoms to select them (Esc clears):

* 2 atoms → *Vector* pierces both; you are prompted for the total length
  (default: interatomic distance plus a short extension on each side).
* 1 atom → *Vector* prompts for the components and optionally a length;
  with no length given the component magnitude is used, otherwise the
  direction is normalised to the given length.
* ≥ 3 atoms → *Plane* (least-squares fit through the atoms); you are
  prompted for the plane width (default: just covering the atoms).
* ≥ 4 atoms → *Polyhedron* (convex hull, as in crystallography software).
* 2 atoms → set the bond type (none/single/double/triple/auto) manually.

Selected atoms are marked with a glowing halo.  The selection is cleared
automatically once an object has been created from it.

Each object becomes a preview cell.  When an object's cell is active, its
properties (color, opacity, length/width/radius) are editable in the
*Object / surface properties* panel at the bottom of the options bar;
`Del` deletes the object.  When an orbital's cell is active, the same
panel shows the isosurface settings instead — isovalue, lobe colors and
opacity — and any change updates all isosurfaces automatically (the
volumes are cached, so only marching cubes re-runs).  `Del` with atoms selected removes the atoms (and their
bonds) from all views —
the underlying file and computed isosurfaces are never modified.
`Ctrl+Z` / `Ctrl+Y` undo and redo.

**Options.**  The *Options* dialog controls representation
(ball-and-stick / sticks / wireframe), atom scale, bond radius,
background color and the per-element colors and radii.  (Hydrogen and
multiple-bond visibility are the *Visible items* toggles; the isovalue,
lobe colors and opacity are set in the properties panel described above,
and default to red for positive and blue for negative lobes.)  Everything
applies live and persists between sessions; *Reset to defaults* restores
the shipped values.  Automatic bonds use the Pyykkö additive covalent
radii (single, double and triple).

**Export.**  Tick the *export* box of the preview cells you want, then
*Export images…*.  Options: output folder, base name, image size as a
multiple of the main-view size (the resulting resolution is shown, with a
lower bound and a memory guard), transparent or colored background
(transparent is the default), the orbital grid spacing used for the
recomputation at export quality, PNG compression level (max by default)
and *crop to content*.  Existing files prompt individually with a *Yes to
all* option.  Orbital images are named `<base><number>[a|b].png` with
zero-padded numbers of uniform width; other images get a running number.
Once started the export runs on its own; a progress bar tracks it.

**Scratch data.**  Orbital volumes are cached in a temporary directory
(shown in the status bar) and the whole directory is deleted when Kaijo
exits.

## Tests

The tests are self-contained: they use a curated set of orbital and
geometry files bundled in `tests/data/` (copied from `example_data/`), so
they run even when `example_data/` is absent.  The set spans ORCA and
Gaussian 16 output, restricted/unrestricted/restricted-open-shell methods,
Pople/def2/x2c bases, and all-electron as well as small-core and
4f-in-core ECP calculations.

```sh
python3 tests/test_math.py     # solid harmonics + orbital norms (all data files)
python3 tests/test_fields.py   # density / cusp / ESP validation
python3 tests/test_fchk.py     # fchk loader, cross-validated against molden
python3 tests/test_render.py   # offscreen render (structure, orbital, geometry)
python3 tests/test_gui_flow.py /tmp   # scripted end-to-end GUI flow
```

Each test also accepts explicit molden paths as arguments to check other
files.
