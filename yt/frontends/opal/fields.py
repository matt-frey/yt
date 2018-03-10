"""
Opal code fields

"""

#-----------------------------------------------------------------------------
# Copyright (c) 2016, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import string
import re

from yt.fields.field_info_container import \
    FieldInfoContainer

spec_finder = re.compile(r'.*\((\D*)(\d*)\).*')

class OpalFieldInfo(FieldInfoContainer):
    known_other_fields = (
        ("rho", ("C/m**3", ["charge_density"], r"\rho")),
        ("Ex", ("V / m", ["electric_field_x"], r"E_{x}")),
        ("Ey", ("V / m", ["electric_field_y"], r"E_{y}")),
        ("Ez", ("V / m", ["electric_field_z"], r"E_{z}")),
        ("phi", ("V", ["electrostatic_potential"], r"\Phi")),
    )

    known_particle_fields = (
        ("particle_mass", ("GeV / c ** 2", [], None)),
        ("particle_charge", ("C", [], None)),
        ("particle_position_x", ("code_length", [], None)),
        ("particle_position_y", ("code_length", [], None)),
        ("particle_position_z", ("code_length", [], None)),
        ("particle_momentum_x", ("bg", [], r"p_{x}")), # r"\beta\gamma"
        ("particle_momentum_y", ("bg", [], r"p_{y}")),
        ("particle_momentum_z", ("bg", [], r"p_{z}")),
        ("particle_timestep", ("code_time", [], None)),
        ("particle_potential", ("V", [], r"\Phi")),
        ("particle_electric_field_x", ("V / m", [], r"E_{x}")),
        ("particle_electric_field_y", ("V / m", [], r"E_{y}")),
        ("particle_electric_field_z", ("V / m", [], r"E_{z}")),
        ("particle_magnetic_field_x", ("T", [], r"B_{x}")),
        ("particle_magnetic_field_y", ("T", [], r"B_{y}")),
        ("particle_magnetic_field_z", ("T", [], r"B_{z}")),
        #("particle_id", ("", ["particle_index"], None)),
        #("particle_cpu", ("", ["particle_index"], None)),
    )
    
    #extra_union_fields = (
        #("GeV/c ** 2", "particle_mass"),
        #("C", "particle_charge"),
    #)
    
    #def setup_particle_fields(self, ptype):
        #super(OpalFieldInfo, self).setup_particle_fields('')
        ###pass
    
    def setup_fluid_fields(self):
        pass
    
    def setup_fluid_index_fields(self):
        pass
    
    #def setup_extra_union_fields(self, ptype="all"):
        #pass
    
    def setup_smoothed_fields(self, ptype, num_neighbors = 64, ftype = "gas"):
        pass
