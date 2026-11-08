"""Materials.

Procedural throughout -- no image textures to ship alongside the model. The
scale pattern is a Voronoi cell pattern driven by object coordinates, which is
what gives a crocodile its irregular plates without a UV layout; the same node
group does denim weave and leather grain at different scales.
"""

from __future__ import annotations

import bpy

from .proportions import P


def _base(name: str, colour, roughness: float = 0.6, metallic: float = 0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = colour
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def _cell_bump(mat, scale: float, strength: float, randomness: float = 1.0):
    """Add a Voronoi-cell bump so a flat colour reads as scales or grain."""
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    coord = nt.nodes.new("ShaderNodeTexCoord")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = scale
    vor.inputs["Randomness"].default_value = randomness
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = 0.02

    nt.links.new(coord.outputs["Object"], vor.inputs["Vector"])
    nt.links.new(vor.outputs["Distance"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return vor, bump


def _two_tone(mat, top_colour, belly_colour, split_z: float = 0.0, blend: float = 0.25):
    """Darken the back and lighten the belly by height in object space."""
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = max(0.0, split_z - blend)
    ramp.color_ramp.elements[0].color = belly_colour
    ramp.color_ramp.elements[1].position = min(1.0, split_z + blend)
    ramp.color_ramp.elements[1].color = top_colour

    nt.links.new(coord.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Y"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return ramp


def build() -> dict[str, bpy.types.Material]:
    mats: dict[str, bpy.types.Material] = {}

    skin = _base("Croc_Skin", P.scale_back, roughness=0.52)
    # Generated coords run 0..1 over the bounding box; -Y is the character's
    # front, so the belly lightens toward the low end of that axis.
    _two_tone(skin, P.scale_back, P.scale_belly, split_z=0.42, blend=0.22)
    _cell_bump(skin, 110.0, 0.42)
    mats["skin"] = skin

    hide = _base("Croc_Hide", P.scale_back, roughness=0.58)
    _cell_bump(hide, 55.0, 0.55)
    mats["hide"] = hide

    shirt = _base("Shirt", P.shirt, roughness=0.88)
    _cell_bump(shirt, 420.0, 0.12, randomness=0.35)
    mats["shirt"] = shirt

    denim = _base("Denim", P.denim, roughness=0.92)
    _cell_bump(denim, 520.0, 0.14, randomness=0.25)
    mats["denim"] = denim

    leather = _base("Leather", P.leather, roughness=0.66)
    _cell_bump(leather, 260.0, 0.30)
    mats["leather"] = leather

    mats["metal"] = _base("Metal", P.metal, roughness=0.34, metallic=0.9)
    mats["bone"] = _base("Bone", P.bone, roughness=0.40)
    mats["claw"] = _base("Claw", P.claw, roughness=0.28)
    mats["mouth"] = _base("Mouth", P.mouth, roughness=0.55)

    gem = _base("Gem", P.gem, roughness=0.08)
    gem.node_tree.nodes["Principled BSDF"].inputs["IOR"].default_value = 1.8
    mats["gem"] = gem

    eye = _base("Eye", P.eye, roughness=0.12)
    mats["eye"] = eye

    return mats
