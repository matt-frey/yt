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
        ("rho", ("C/m**3", ["rho"], r"\rho")),
        ("Ex", ("V / m", ["Ex"], r"E_{x}")),
        ("Ey", ("V / m", ["Ey"], r"E_{y}")),
        ("Ez", ("V / m", ["Ez"], r"E_{z}")),
        ("potential", ("V", ["potential"], r"\Phi"))
    )

    known_particle_fields = (
        ("particle_mass", ("eV / c ** 2", [], None)),
        ("particle_charge", ("code_charge", [], None)),
        ("particle_position_x", ("code_length", [], None)),
        ("particle_position_y", ("code_length", [], None)),
        ("particle_position_z", ("code_length", [], None)),
        ("particle_momentum_x", (r"\beta\gamma", [], r"p_{x}")),
        ("particle_momentum_y", (r"\beta\gamma", [], r"p_{y}")),
        ("particle_momentum_z", (r"\beta\gamma", [], r"p_{z}"))
    )
    
    def setup_particle_fields(self, ptype):
        pass
    
    def setup_fluid_fields(self):
        pass
    
    def setup_fluid_index_fields(self):
        pass
    
    def setup_extra_union_fields(self, ptype="all"):
        pass
    
    def setup_smoothed_fields(self, ptype, num_neighbors = 64, ftype = "gas"):
        pass


class OpalSingleFieldInfo(FieldInfoContainer):
    
    known_other_fields = ()
    
    known_particle_fields = ()