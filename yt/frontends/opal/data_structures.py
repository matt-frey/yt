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
    BoxlibHierarchy


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

        #is_checkpoint = True
        #for ptype in self.ds.particle_types:
            #self._read_particles(ptype, is_checkpoint)
        
        ## Additional Opal particle information (used to set up species)
        #self.opal_header = OpalHeader(self.ds.output_dir + "/OpalAmrHeader")
        
        #i = 0
        #for key, val in self.opal_header.data.items():
            #if key.startswith("species_"):
                #charge_name = 'particle%.1d_charge' % i
                #mass_name = 'particle%.1d_mass' % i
                #self.parameters[charge_name] = val[0]
                #self.parameters[mass_name] = val[1]
                #i = i + 1


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
                
        ## set the periodicity based on the integer BC runtime parameters
        #is_periodic = self.parameters['geometry.is_periodic'].split()
        #periodicity = [bool(val) for val in is_periodic]
        #for _ in range(self.dimensionality, 3):
            #periodicity += [True]  # pad to 3D
        #self.periodicity = ensure_tuple(periodicity)

        

        #species_list = []
        #species_dirs = glob.glob(self.output_dir + "/particle*")
        #for species in species_dirs:
            #species_list.append(species[len(self.output_dir)+1:])
        #self.particle_types = tuple(species_list)
        #self.particle_types_raw = self.particle_types


    def _set_code_unit_attributes(self):
        setdefaultattr(self, 'length_unit', self.quan(1.0, "m"))
        setdefaultattr(self, 'mass_unit', self.quan(1.0, "kg"))
        setdefaultattr(self, 'time_unit', self.quan(1.0, "s"))
        setdefaultattr(self, 'velocity_unit', self.quan(1.0, "m/s"))

#class OpalGrid(AMRGridPatch):
    #_id_offset = 0
    #_offset = -1

    #def __init__(self, grid_id, offset, filename=None,
                 #index=None):
        #super(OpalGrid, self).__init__(grid_id, filename, index)
        #self._base_offset = offset
        #self._parent_id = []
        #self._children_ids = []

    #def _prepare_grid(self):
        #super(OpalGrid, self)._prepare_grid()
        #my_ind = self.id - self._id_offset
        #self.start_index = self.index.grid_start_index[my_ind]

    #def get_global_startindex(self):
        #return self.start_index

    #def _setup_dx(self):
        ## has already been read in and stored in index
        #self.dds = self.index.ds.arr(self.index.level_dds[self.Level, :], 'code_length')
        #self.field_data['dx'], self.field_data['dy'], self.field_data['dz'] = self.dds

    #def __repr__(self):
        #return "OpalGrid_%04i" % (self.id)

    #@property
    #def Parent(self):
        #if len(self._parent_id) == 0:
            #return None
        #return [self.index.grids[pid - self._id_offset]
                #for pid in self._parent_id]

    #@property
    #def Children(self):
        #return [self.index.grids[cid - self._id_offset]
                #for cid in self._children_ids]

    #def _get_offset(self, f):
        ## This will either seek to the _offset or figure out the correct
        ## _offset.
        #if self._offset == -1:
            #f.seek(self._base_offset, os.SEEK_SET)
            #f.readline()
            #self._offset = f.tell()
        #return self._offset

    ## We override here because we can have varying refinement levels
    #def select_ires(self, dobj):
        #mask = self._get_selector_mask(dobj.selector)
        #if mask is None: return np.empty(0, dtype='int64')
        #coords = np.empty(self._last_count, dtype='int64')
        #coords[:] = self.Level + self.ds.level_offsets[self.Level]
        #return coords

    ## Override this as well, since refine_by can vary
    #def _fill_child_mask(self, child, mask, tofill, dlevel=1):
        #rf = self.ds.ref_factors[self.Level]
        #if dlevel != 1:
            #raise NotImplementedError
        #gi, cgi = self.get_global_startindex(), child.get_global_startindex()
        #startIndex = np.maximum(0, cgi // rf - gi)
        #endIndex = np.minimum((cgi + child.ActiveDimensions) // rf - gi,
                              #self.ActiveDimensions)
        #endIndex += (startIndex == endIndex)
        #mask[startIndex[0]:endIndex[0],
             #startIndex[1]:endIndex[1],
             #startIndex[2]:endIndex[2]] = tofill


#class OpalHierarchy(GridIndex):
    #grid = OpalGrid

    #def __init__(self, ds, dataset_type='opal'):
        #self.dataset_type = dataset_type
        #self.header_filename = os.path.join(ds.output_dir, 'OpalAmrHeader')
        #self.directory = ds.output_dir

        #GridIndex.__init__(self, ds, dataset_type)
        #self._cache_endianness(self.grids[-1])

    #def _parse_index(self):
        #"""
        #read the global header file for an Opal plotfile output.
        #"""
        #self.max_level = self.dataset._max_level
        #header_file = open(self.header_filename, 'r')

        #self.dimensionality = self.dataset.dimensionality
        #_our_dim_finder = _dim_finder[self.dimensionality-1]
        #DRE = self.dataset.domain_right_edge  # shortcut
        #DLE = self.dataset.domain_left_edge   # shortcut

        ## We can now skip to the point in the file we want to start parsing.
        #header_file.seek(self.dataset._header_mesh_start)

        #dx = []
        #for i in range(self.max_level + 1):
            #dx.append([float(v) for v in next(header_file).split()])
            ## account for non-3d data sets
            #if self.dimensionality < 2:
                #dx[i].append(DRE[1] - DLE[1])
            #if self.dimensionality < 3:
                #dx[i].append(DRE[2] - DLE[1])
        #self.level_dds = np.array(dx, dtype="float64")
        #next(header_file)
        #if self.ds.geometry == "cartesian":
            #default_ybounds = (0.0, 1.0)
            #default_zbounds = (0.0, 1.0)
        #else:
            #raise RuntimeError("Unknown Opal coordinate system.")
        #if int(next(header_file)) != 0:
            #raise RuntimeError("INTERNAL ERROR! This should be a zero.")

        ## each level is one group with ngrids on it.
        ## each grid has self.dimensionality number of lines of 2 reals
        #self.grids = []
        #grid_counter = 0
        #for level in range(self.max_level + 1):
            #vals = next(header_file).split()
            #lev, ngrids = int(vals[0]), int(vals[1])
            #assert(lev == level)
            #nsteps = int(next(header_file))  # NOQA
            #for gi in range(ngrids):
                #xlo, xhi = [float(v) for v in next(header_file).split()]
                #if self.dimensionality > 1:
                    #ylo, yhi = [float(v) for v in next(header_file).split()]
                #else:
                    #ylo, yhi = default_ybounds
                #if self.dimensionality > 2:
                    #zlo, zhi = [float(v) for v in next(header_file).split()]
                #else:
                    #zlo, zhi = default_zbounds
                #self.grid_left_edge[grid_counter + gi, :] = [xlo, ylo, zlo]
                #self.grid_right_edge[grid_counter + gi, :] = [xhi, yhi, zhi]
            ## Now we get to the level header filename, which we open and parse.
            #fn = os.path.join(self.dataset.output_dir,
                              #next(header_file).strip())
            #level_header_file = open(fn + "_H")
            #level_dir = os.path.dirname(fn)
            ## We skip the first two lines, which contain Opal header file
            ## version and 'how' the data was written
            #next(level_header_file)
            #next(level_header_file)
            ## Now we get the number of components
            #ncomp_this_file = int(next(level_header_file))  # NOQA
            ## Skip the next line, which contains the number of ghost zones
            #next(level_header_file)
            ## To decipher this next line, we expect something like:
            ## (8 0
            ## where the first is the number of FABs in this level.
            #ngrids = int(next(level_header_file).split()[0][1:])
            ## Now we can iterate over each and get the indices.
            #for gi in range(ngrids):
                ## components within it
                #start, stop = _our_dim_finder.match(next(level_header_file)).groups()
                ## fix for non-3d data 
                ## note we append '0' to both ends b/c of the '+1' in dims below
                #start += ',0'*(3-self.dimensionality)
                #stop += ',0'*(3-self.dimensionality)
                #start = np.array(start.split(","), dtype="int64")
                #stop = np.array(stop.split(","), dtype="int64")
                #dims = stop - start + 1
                #self.grid_dimensions[grid_counter + gi,:] = dims
                #self.grid_start_index[grid_counter + gi,:] = start
            ## Now we read two more lines.  The first of these is a close
            ## parenthesis.
            #next(level_header_file)
            ## The next is again the number of grids
            #next(level_header_file)
            ## Now we iterate over grids to find their offsets in each file.
            #for gi in range(ngrids):
                ## Now we get the data file, at which point we're ready to
                ## create the grid.
                #dummy, filename, offset = next(level_header_file).split()
                #filename = os.path.join(level_dir, filename)
                #go = self.grid(grid_counter + gi, int(offset), filename, self)
                #go.Level = self.grid_levels[grid_counter + gi,:] = level
                #self.grids.append(go)
            #grid_counter += ngrids
            ## already read the filenames above...
        #self.float_type = 'float64'

    #def _cache_endianness(self, test_grid):
        #"""
        #Cache the endianness and bytes perreal of the grids by using a
        #test grid and assuming that all grids have the same
        #endianness. This is a pretty safe assumption since Opal uses
        #one file per processor, and if you're running on a cluster
        #with different endian processors, then you're on your own!
        #"""
        ## open the test file & grab the header
        #with open(os.path.expanduser(test_grid.filename), 'rb') as f:
            #header = f.readline().decode("ascii", "ignore")

        #bpr, endian, start, stop, centering, nc = \
            #_header_pattern[self.dimensionality-1].search(header).groups()
        ## Note that previously we were using a different value for BPR than we
        ## use now.  Here is an example set of information directly from Opal:
        ##  * DOUBLE data
        ##  * FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (1 2 3 4 5 6 7 8)))((0,0) (63,63) (0,0)) 27
        ##  * FLOAT data
        ##  * FAB ((8, (32 8 23 0 1 9 0 127)),(4, (1 2 3 4)))((0,0) (63,63) (0,0)) 27
        #if bpr == endian[0]:
            #dtype = '<f%s' % bpr
        #elif bpr == endian[-1]:
            #dtype = '>f%s' % bpr
        #else:
            #raise ValueError("FAB header is neither big nor little endian. Perhaps the file is corrupt?")

        #mylog.debug("FAB header suggests dtype of %s", dtype)
        #self._dtype = np.dtype(dtype)

    #def _populate_grid_objects(self):
        #mylog.debug("Creating grid objects")
        #self.grids = np.array(self.grids, dtype='object')
        #self._reconstruct_parent_child()
        #for i, grid in enumerate(self.grids):
            #if (i % 1e4) == 0: mylog.debug("Prepared % 7i / % 7i grids", i,
                                           #self.num_grids)
            #grid._prepare_grid()
            #grid._setup_dx()
        #mylog.debug("Done creating grid objects")

    #def _reconstruct_parent_child(self):
        #mask = np.empty(len(self.grids), dtype='int32')
        #mylog.debug("First pass; identifying child grids")
        #for i, grid in enumerate(self.grids):
            #get_box_grids_level(self.grid_left_edge[i,:],
                                #self.grid_right_edge[i,:],
                                #self.grid_levels[i] + 1,
                                #self.grid_left_edge, self.grid_right_edge,
                                #self.grid_levels, mask)
            #ids = np.where(mask.astype("bool"))  # where is a tuple
            #grid._children_ids = ids[0] + grid._id_offset
        #mylog.debug("Second pass; identifying parents")
        #for i, grid in enumerate(self.grids):  # Second pass
            #for child in grid.Children:
                #child._parent_id.append(i + grid._id_offset)

    #def _count_grids(self):
        ## We can get everything from the OpalAmrHeader file, but note that we're
        ## duplicating some work done elsewhere.  In a future where we don't
        ## pre-allocate grid arrays, this becomes unnecessary.
        #header_file = open(self.header_filename, 'r')
        #header_file.seek(self.dataset._header_mesh_start)
        ## Skip over the level dxs, geometry and the zero:
        #[next(header_file) for i in range(self.dataset._max_level + 3)]
        ## Now we need to be very careful, as we've seeked, and now we iterate.
        ## Does this work?  We are going to count the number of places that we
        ## have a three-item line.  The three items would be level, number of
        ## grids, and then grid time.
        #self.num_grids = 0
        #for line in header_file:
            #if len(line.split()) != 3: continue
            #self.num_grids += int(line.split()[1])

    #def _initialize_grid_arrays(self):
        #super(OpalHierarchy, self)._initialize_grid_arrays()
        #self.grid_start_index = np.zeros((self.num_grids, 3), 'int64')

    #def _initialize_state_variables(self):
        #"""override to not re-initialize num_grids in AMRHierarchy.__init__

        #"""
        #self._parallel_locking = False
        #self._data_file = None
        #self._data_mode = None

    #def _detect_output_fields(self):
        ## This is all done in _parse_header_file
        #self.field_list = [("opal", f) for f in
                           #self.dataset._field_list]
        #self.field_indexes = dict((f[1], i)
                                  #for i, f in enumerate(self.field_list))
        ## There are times when field_list may change.  We copy it here to
        ## avoid that possibility.
        #self.field_order = [f for f in self.field_list]

    #def _setup_data_io(self):
        #self.io = io_registry[self.dataset_type](self.dataset)


#class OpalDataset(Dataset):
    #"""
    #This class is a stripped down class that simply reads and parses
    #*filename*, without looking at the Opal index.
    #"""
    #_index_class = OpalHierarchy
    #_field_info_class = OpalFieldInfo
    #_output_prefix = None

    ## THIS SHOULD BE FIXED:
    #periodicity = (True, True, True)

    #def __init__(self, output_dir,
                 #dataset_type='opal',
                 #storage_filename=None,
                 #units_override=None,
                 #unit_system="mks"):
        #"""
        #Do not remove self.fluid_types otherwise data is not recognized
        #"""
        #self.fluid_types += ("opal",)
        #self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        #self.storage_filename = storage_filename

        #Dataset.__init__(self, output_dir, dataset_type,
                         #units_override=units_override,
                         #unit_system=unit_system)

    #def _localize_check(self, fn):
        ## If the file exists, use it.  If not, set it to None.
        #root_dir = os.path.dirname(self.output_dir)
        #full_fn = os.path.join(root_dir, fn)
        #if os.path.exists(full_fn):
            #return full_fn
        #return None

    #@classmethod
    #def _is_valid(cls, *args, **kwargs):
        ## fill our args
        #output_dir = args[0]
        ## opal datasets are always directories
        #if not os.path.isdir(output_dir): return False
        #header_filename = os.path.join(output_dir, "OpalAmrHeader")
        
        #if not os.path.isfile(header_filename):
            #return False
        
        #print(header_filename)
        #if not os.path.exists(header_filename):
            ## We *know* it's not opal if OpalAmrHeader doesn't exist.
            #return False
        #args = inspect.getcallargs(cls.__init__, args, kwargs)
        
        #return True

    #def _parse_parameter_file(self):
        #"""
        #Parses the parameter file and establishes the various
        #dictionaries.
        #"""
        #self._parse_header_file()
        ## Let's read the file
        #hfn = os.path.join(self.output_dir, 'OpalAmrHeader')
        #self.unique_identifier = int(os.stat(hfn)[ST_CTIME])

    #def _parse_header_file(self):
        #"""
        #We parse the Opal header, which we use as our basis.  Anything in the
        #inputs file will override this, but the inputs file is not strictly
        #necessary for orientation of the data in space.
        #"""

        ## Note: Python uses a read-ahead buffer, so using next(), which would
        ## be my preferred solution, won't work here.  We have to explicitly
        ## call readline() if we want to end up with an offset at the very end.
        ## Fortunately, elsewhere we don't care about the offset, so we're fine
        ## everywhere else using iteration exclusively.
        #header_file = open(os.path.join(self.output_dir, 'OpalAmrHeader'))
        #self.opal_version = header_file.readline().rstrip()
        #n_fields = int(header_file.readline())

        #self._field_list = [header_file.readline().strip()
                            #for i in range(n_fields)]

        #self.dimensionality = int(header_file.readline())
        #self.current_time = float(header_file.readline())
        ## This is traditionally a index attribute, so we will set it, but
        ## in a slightly hidden variable.
        #self._max_level = int(header_file.readline())
        #self.domain_left_edge = np.array(header_file.readline().split(),
                                         #dtype="float64")
        #self.domain_right_edge = np.array(header_file.readline().split(),
                                          #dtype="float64")
        #ref_factors = np.array([int(i) for i in
                                #header_file.readline().split()])
        #if ref_factors.size == 0:
            ## We use a default of two, as Nyx doesn't always output this value
            #ref_factors = [2] * (self._max_level + 1)
        ## We can't vary refinement factors based on dimension, or whatever else
        ## they are vaied on.  In one curious thing, I found that some Castro 3D
        ## data has only two refinement factors, which I don't know how to
        ## understand.
        #self.ref_factors = ref_factors
        #if np.unique(ref_factors).size > 1:
            ## We want everything to be a multiple of this.
            #self.refine_by = min(ref_factors)
            ## Check that they're all multiples of the minimum.
            #if not all(float(rf)/self.refine_by ==
                       #int(float(rf)/self.refine_by) for rf in ref_factors):
                #raise RuntimeError
            #base_log = np.log2(self.refine_by)
            #self.level_offsets = [0]  # level 0 has to have 0 offset
            #lo = 0
            #for lm1, rf in enumerate(self.ref_factors):
                #lo += int(np.log2(rf) / base_log) - 1
                #self.level_offsets.append(lo)
        ## assert(np.unique(ref_factors).size == 1)
        #else:
            #self.refine_by = ref_factors[0]
            #self.level_offsets = [0 for l in range(self._max_level + 1)]
        ## Now we read the global index space, to get
        #index_space = header_file.readline()
        ## This will be of the form:
        ##  ((0,0,0) (255,255,255) (0,0,0)) ((0,0,0) (511,511,511) (0,0,0))
        ## So note that if we split it all up based on spaces, we should be
        ## fine, as long as we take the first two entries, which correspond to
        ## the root level.  I'm not 100% pleased with this solution.
        #root_space = index_space.replace("(", "").replace(")", "").split()[:2]
        #start = np.array(root_space[0].split(","), dtype="int64")
        #stop = np.array(root_space[1].split(","), dtype="int64")
        #self.domain_dimensions = stop - start + 1
        ## Skip timesteps per level
        #header_file.readline()
        #self._header_mesh_start = header_file.tell()
        ## Skip the cell size information per level - we'll get this later
        #for i in range(self._max_level+1): header_file.readline()
        ## Get the geometry
        #next_line = header_file.readline()
        #if len(next_line.split()) == 1:
            #coordinate_type = int(next_line)
        #else:
            #coordinate_type = 0
        #if coordinate_type == 0:
            #self.geometry = "cartesian"
        #else:
            #raise RuntimeError("Unknown Opal coord_type")

    #def _set_code_unit_attributes(self):
        #setdefaultattr(self, 'length_unit', self.quan(1.0, "m"))
        #setdefaultattr(self, 'mass_unit', self.quan(1.0, "eV / c ** 2"))
        #setdefaultattr(self, 'time_unit', self.quan(1.0, "s"))
        #setdefaultattr(self, 'velocity_unit', self.quan(1.0, "m/s"))
        ##setdefaultattr(self, 'momentum_unit', self.quan(1.0, ""))

    #@parallel_root_only
    #def print_key_parameters(self):
        #for a in ["current_time", "domain_dimensions", "domain_left_edge",
                  #"domain_right_edge"]:
            #if not hasattr(self, a):
                #mylog.error("Missing %s in parameter file definition!", a)
                #continue
            #v = getattr(self, a)
            #mylog.info("Parameters: %-25s = %s", a, v)

    #def relative_refinement(self, l0, l1):
        #offset = self.level_offsets[l1] - self.level_offsets[l0]
        #return self.refine_by**(l1-l0 + offset)

## --------------------------------------------------------------------------------------------



#class OpalSingleGrid(AMRGridPatch):
    
    #def __init__(self, grid_id, offset, filename=None,
                 #index=None):
        #super(OpalSingleGrid, self).__init__(grid_id, filename, index)
        #self._base_offset = offset
        #self._parent_id = []
        #self._children_ids = []


#class OpalSingleHierarchy(GridIndex):
    #grid = OpalSingleGrid

    #def __init__(self, ds, dataset_type='opal_native'):
        #print ("OpalSingleHierarchy")
        #self.dataset_type = dataset_type
        #self.directory = ds.output_dir

#class OpalSingleDataset(Dataset):
    #"""
    #"""
    #_index_class = OpalSingleHierarchy
    #_field_info_class = OpalSingleFieldInfo
    ##_output_prefix = None

    ## 2. called
    #def __init__(self, output_dir,
                 #dataset_type='opal_native',
                 #storage_filename=None,
                 #units_override=None,
                 #unit_system="mks"):
        
        #print ("OpalSingleDataset")
        #self.fluid_types += ("opal",)
        #self.current_time = 0.0
        #self.unique_identifier = 'bla' #int(os.stat(hfn)[ST_CTIME])
        #self.refine_by = 2
        #self.dimensionality = 3
        
        #self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        ##self.storage_filename = storage_filename

        #Dataset.__init__(self, output_dir, dataset_type,
                         #units_override=units_override,
                         #unit_system=unit_system)
    
    
    ## 1. called
    #@classmethod
    #def _is_valid(cls, *args, **kwargs):
        #return False
    
    ## 3. called
    #def _parse_parameter_file(self):
        
        #self.cosmological_simulation = 0
        
        
        #print("Filename: ", self.parameter_filename)
        #self._handle = h5py.File(self.parameter_filename, "r")
        
    
    ## 4. called
    #def _set_code_unit_attributes(self):
        #setdefaultattr(self, 'length_unit', self.quan(1.0, "m"))
        #setdefaultattr(self, 'mass_unit', self.quan(1.0, "eV / c ** 2"))
        #setdefaultattr(self, 'time_unit', self.quan(1.0, "s"))
        #setdefaultattr(self, 'velocity_unit', self.quan(1.0, "m/s"))
        ##setdefaultattr(self, 'momentum_unit', self.quan(1.0, ""))
        
        #self._parse_unit_attributes()
    
    
    ## own functionalities
    #def _parse_unit_attributes(self):
        
        #attrs = self._handle["/"].attrs
        
        #emittance_unit = attrs.get("#varepsilonUnit")
        #print ( emittance_unit.value )
        