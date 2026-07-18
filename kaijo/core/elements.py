# SPDX-License-Identifier: GPL-3.0-or-later
"""Element data: symbols, colors, covalent radii (Pyykkö), display radii.

Radii are in Ångström.  Covalent radii are the Pyykkö additive radii:
  single bond: P. Pyykkö, M. Atsumi, Chem. Eur. J. 15 (2009) 186.
  double bond: P. Pyykkö, M. Atsumi, Chem. Eur. J. 15 (2009) 12770.
  triple bond: P. Pyykkö, S. Riedel, M. Patzschke, Chem. Eur. J. 11 (2005) 3511.
Colors follow the Jmol/CPK convention.
"""

BOHR = 0.529177210903  # Å per bohr

SYMBOLS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

_SYM_TO_Z = {s.lower(): z for z, s in enumerate(SYMBOLS)}


def symbol_to_z(symbol):
    """Element symbol -> atomic number (0 if unknown)."""
    return _SYM_TO_Z.get(symbol.strip().lower(), 0)


def z_to_symbol(z):
    if 0 < z < len(SYMBOLS):
        return SYMBOLS[z]
    return "X"


# Pyykkö single-bond covalent radii in pm, indexed by Z (0 unused).
_R1_PM = [
    0, 32, 46, 133, 102, 85, 75, 71, 63, 64, 67,
    155, 139, 126, 116, 111, 103, 99, 96, 196, 171,
    148, 136, 134, 122, 119, 116, 111, 110, 112, 118,
    124, 121, 121, 116, 114, 117, 210, 185, 163, 154,
    147, 138, 128, 125, 125, 120, 128, 136, 142, 140,
    140, 136, 133, 131, 232, 196, 180, 163, 176, 174,
    173, 172, 168, 169, 168, 167, 166, 165, 164, 170,
    162, 152, 146, 137, 131, 129, 122, 123, 124, 133,
    144, 144, 151, 145, 147, 142, 223, 201, 186, 175,
    169, 170, 171, 172, 166, 166, 168, 168, 165, 167,
    173, 176, 161, 157, 149, 143, 141, 134, 129, 128,
    121, 122, 136, 143, 162, 175, 165, 157,
]

# Pyykkö double-bond covalent radii in pm (0 where not defined).
_R2_PM = [
    0, 0, 0, 124, 90, 78, 67, 60, 57, 59, 96,
    160, 132, 113, 107, 102, 94, 95, 107, 193, 147,
    116, 117, 112, 111, 105, 109, 103, 101, 115, 120,
    117, 111, 114, 107, 109, 121, 202, 157, 130, 127,
    125, 121, 120, 114, 110, 117, 139, 144, 136, 130,
    133, 128, 129, 135, 209, 161, 139, 137, 138, 137,
    135, 134, 134, 135, 135, 133, 133, 133, 131, 129,
    131, 128, 126, 120, 119, 116, 115, 112, 121, 142,
    142, 135, 141, 135, 138, 145, 218, 173, 153, 143,
    138, 134, 136, 135, 135, 136, 139, 140, 140, 0,
    139, 0, 141, 140, 136, 128, 128, 125, 125, 116,
    116, 137, 0, 0, 0, 0, 0, 0,
]

# Pyykkö triple-bond covalent radii in pm (0 where not defined).
_R3_PM = [
    0, 0, 0, 0, 85, 73, 60, 54, 53, 53, 0,
    0, 127, 111, 102, 94, 95, 93, 96, 0, 133,
    114, 108, 106, 103, 103, 102, 96, 101, 120, 0,
    121, 114, 106, 107, 110, 108, 0, 139, 124, 121,
    116, 113, 110, 103, 106, 112, 137, 0, 146, 132,
    127, 121, 125, 122, 0, 149, 139, 131, 128, 0,
    0, 0, 0, 132, 0, 0, 0, 0, 0, 0,
    131, 122, 119, 115, 110, 109, 107, 110, 123, 0,
    150, 137, 135, 129, 138, 133, 0, 159, 140, 136,
    129, 118, 116, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 131, 126, 121, 119, 118, 113, 112,
    118, 130, 0, 0, 0, 0, 0, 0,
]

COVALENT_R1 = [r / 100.0 for r in _R1_PM]
COVALENT_R2 = [r / 100.0 if r else None for r in _R2_PM]
COVALENT_R3 = [r / 100.0 if r else None for r in _R3_PM]

# Jmol element colors, (r, g, b) in 0..1, indexed by Z.
_JMOL_HEX = {
    0: 0xFF1493, 1: 0xFFFFFF, 2: 0xD9FFFF, 3: 0xCC80FF, 4: 0xC2FF00,
    5: 0xFFB5B5, 6: 0x909090, 7: 0x3050F8, 8: 0xFF0D0D, 9: 0x90E050,
    10: 0xB3E3F5, 11: 0xAB5CF2, 12: 0x8AFF00, 13: 0xBFA6A6, 14: 0xF0C8A0,
    15: 0xFF8000, 16: 0xFFFF30, 17: 0x1FF01F, 18: 0x80D1E3, 19: 0x8F40D4,
    20: 0x3DFF00, 21: 0xE6E6E6, 22: 0xBFC2C7, 23: 0xA6A6AB, 24: 0x8A99C7,
    25: 0x9C7AC7, 26: 0xE06633, 27: 0xF090A0, 28: 0x50D050, 29: 0xC88033,
    30: 0x7D80B0, 31: 0xC28F8F, 32: 0x668F8F, 33: 0xBD80E3, 34: 0xFFA100,
    35: 0xA62929, 36: 0x5CB8D1, 37: 0x702EB0, 38: 0x00FF00, 39: 0x94FFFF,
    40: 0x94E0E0, 41: 0x73C2C9, 42: 0x54B5B5, 43: 0x3B9E9E, 44: 0x248F8F,
    45: 0x0A7D8C, 46: 0x006985, 47: 0xC0C0C0, 48: 0xFFD98F, 49: 0xA67573,
    50: 0x668080, 51: 0x9E63B5, 52: 0xD47A00, 53: 0x940094, 54: 0x429EB0,
    55: 0x57178F, 56: 0x00C900, 57: 0x70D4FF, 58: 0xFFFFC7, 59: 0xD9FFC7,
    60: 0xC7FFC7, 61: 0xA3FFC7, 62: 0x8FFFC7, 63: 0x61FFC7, 64: 0x45FFC7,
    65: 0x30FFC7, 66: 0x1FFFC7, 67: 0x00FF9C, 68: 0x00E675, 69: 0x00D452,
    70: 0x00BF38, 71: 0x00AB24, 72: 0x4DC2FF, 73: 0x4DA6FF, 74: 0x2194D6,
    75: 0x267DAB, 76: 0x266696, 77: 0x175487, 78: 0xD0D0E0, 79: 0xFFD123,
    80: 0xB8B8D0, 81: 0xA6544D, 82: 0x575961, 83: 0x9E4FB5, 84: 0xAB5C00,
    85: 0x754F45, 86: 0x428296, 87: 0x420066, 88: 0x007D00, 89: 0x70ABFA,
    90: 0x00BAFF, 91: 0x00A1FF, 92: 0x008FFF, 93: 0x0080FF, 94: 0x006BFF,
    95: 0x545CF2, 96: 0x785CE3, 97: 0x8A4FE3, 98: 0xA136D4, 99: 0xB31FD4,
    100: 0xB31FBA, 101: 0xB30DA6, 102: 0xBD0D87, 103: 0xC70066,
}


def element_color(z):
    """Default color (r, g, b) in 0..1 for atomic number z."""
    h = _JMOL_HEX.get(z, 0xFF1493)
    return ((h >> 16 & 0xFF) / 255.0, (h >> 8 & 0xFF) / 255.0, (h & 0xFF) / 255.0)


def display_radius(z):
    """Ball radius (Å) for the ball-and-stick representation.

    Derived from the covalent radius but compressed so atoms do not differ
    excessively in size, as per the design spec.
    """
    r = COVALENT_R1[z] if 0 < z < len(COVALENT_R1) else 0.75
    return min(max(0.224 + 0.224 * r, 0.26), 0.64)


def covalent_r1(z):
    return COVALENT_R1[z] if 0 < z < len(COVALENT_R1) else 1.5


def covalent_r2(z):
    return COVALENT_R2[z] if 0 < z < len(COVALENT_R2) else None


def covalent_r3(z):
    return COVALENT_R3[z] if 0 < z < len(COVALENT_R3) else None
