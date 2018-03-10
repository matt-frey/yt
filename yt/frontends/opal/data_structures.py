"""
Data structures for Opal Codes



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2016, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import inspect
import os
import re

from stat import ST_CTIME


from yt.utilities.on_demand_imports import _h5py as h5py
import numpy as np

from yt.funcs import \
    ensure_tuple, \
    mylog, \
    setdefaultattr
from yt.data_objects.grid_patch import AMRGridPatch
from yt.extern.six.moves import zip as izip
from yt.geometry.grid_geometry_handler import GridIndex
from yt.data_objects.static_output import Dataset

from yt.utilities.parallel_tools.parallel_analysis_interface import \
    parallel_root_only
from yt.utilities.lib.misc_utilities import \
    get_box_grids_level
from yt.utilities.io_handler import \
    io_registry

from .fields import \
    OpalFieldInfo

from yt.frontends.boxlib.data_structures import \
    BoxlibFieldInfo, \
    BoxlibGrid, \
    BoxlibDataset, \
    BoxlibHierarchy, \
    AMReXParticleHeader

# This is what we use to find scientific notation that might include d's
# instead of e's.
_scinot_finder = re.compile(r"[-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?")
# This is the dimensions in the Cell_H file for each level
# It is different for different dimensionalities, so we make a list
_dim_finder = [
    re.compile(r"\(\((\d+)\) \((\d+)\) \(\d+\)\)$"),
    re.compile(r"\(\((\d+,\d+)\) \((\d+,\d+)\) \(\d+,\d+\)\)$"),
    re.compile(r"\(\((\d+,\d+,\d+)\) \((\d+,\d+,\d+)\) \(\d+,\d+,\d+\)\)$")]
# This is the line that prefixes each set of data for a FAB in the FAB file
# It is different for different dimensionalities, so we make a list
_endian_regex = r"^FAB \(\(\d+, \([0-9 ]+\)\),\((\d+), \(([0-9 ]+)\)\)\)"
_header_pattern = [
    re.compile(_endian_regex +
               r"\(\((\d+)\) \((\d+)\) \((\d+)\)\) (\d+)\n"),
    re.compile(_endian_regex +
               r"\(\((\d+,\d+)\) \((\d+,\d+)\) \((\d+,\d+)\)\) (\d+)\n"),
    re.compile(_endian_regex +
               r"\(\((\d+,\d+,\d+)\) \((\d+,\d+,\d+)\) \((\d+,\d+,\d+)\)\) (\d+)\n")]


class OpalHierarchy(BoxlibHierarchy):

    def __init__(self, ds, dataset_type="boxlib_opal"):
        super(OpalHierarchy, self).__init__(ds, dataset_type)
        
        if ("particles" in self.ds.parameters):
            is_checkpoint = True
            for ptype in self.ds.particle_types:
                self._read_particles(ptype, is_checkpoint)
    
    def _determine_particle_output_type(self, directory_name):
        header_filename =  self.ds.output_dir + "/" + directory_name + "/Header"
        with open(header_filename, "r") as f:
            version_string = f.readline().strip()
            if not version_string.startswith("Version_Two"):
                raise RuntimeError("It's not Version_Two");
            return OpalParticleHeader


class OpalDataset(BoxlibDataset):

    _index_class = OpalHierarchy
    _field_info_class = OpalFieldInfo

    def __init__(self, output_dir,
                 cparam_filename=None,
                 fparam_filename=None,
                 dataset_type='boxlib_opal',
                 storage_filename=None,
                 units_override=None,
                 unit_system="mks"):

        super(OpalDataset, self).__init__(output_dir,
                                           cparam_filename,
                                           fparam_filename,
                                           dataset_type,
                                           storage_filename,
                                           units_override,
                                           unit_system)

    @classmethod
    def _is_valid(cls, *args, **kwargs):
        # fill our args
        output_dir = args[0]
        # boxlib datasets are always directories
        if not os.path.isdir(output_dir): return False
        header_filename = os.path.join(output_dir, "Header")
        if os.path.exists(header_filename):
            return True
        return False

    def _parse_parameter_file(self):
        super(OpalDataset, self)._parse_parameter_file()
        
        if os.path.isdir(os.path.join(self.output_dir, "opal")):
            # we have particles
            self.parameters["particles"] = 1
            self.particle_types = ("opal",)
            self.particle_types_raw =self.particle_types

    def _set_code_unit_attributes(self):
        setdefaultattr(self, 'length_unit', self.quan(1.0, "m"))
        setdefaultattr(self, 'mass_unit', self.quan(1.0, "kg"))
        setdefaultattr(self, 'time_unit', self.quan(1.0, "s"))
        setdefaultattr(self, 'velocity_unit', self.quan(1.0, "m/s"))


class OpalParticleHeader(AMReXParticleHeader):
    
    def __init__(self, ds, directory_name, is_checkpoint,
                 extra_field_names=None):
        super(OpalParticleHeader, self).__init__(ds, directory_name,
                                                 is_checkpoint, extra_field_names)
