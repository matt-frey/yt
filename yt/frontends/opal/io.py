"""
OPAL data-file handling functions



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2016, PSI
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import os
import numpy as np
from collections import defaultdict

from yt.utilities.io_handler import \
    BaseIOHandler
from yt.funcs import mylog
from yt.utilities.on_demand_imports import _h5py as h5py

from yt.frontends.boxlib.io import IOHandlerBoxlib


class IOHandlerOpal(IOHandlerBoxlib):
    
    _dataset_type = "boxlib_opal"
    
    _particle_filename = None

    def __init__(self, ds, *args, **kwargs):
        super(IOHandlerOpal, self).__init__(ds, args, kwargs)

    #def _read_fluid_selection(self, chunks, selector, fields, size):
        #if any((ftype != "opal" for ftype, fname in fields)):
            #raise NotImplementedError
        
        #super(IOHandlerOpal, self)._read_fluid_selection(chunks, selector, fields, size)

#class IOHandlerOpal(BaseIOHandler):
    
    #_dataset_type = "opal"
    
    #_particle_filename = None

    #def __init__(self, ds, *args, **kwargs):
        #self.ds = ds

    #def _read_fluid_selection(self, chunks, selector, fields, size):
        #chunks = list(chunks)
        #if any((ftype != "opal" for ftype, fname in fields)):
            #raise NotImplementedError
        #rv = {}
        #for field in fields:
            #rv[field] = np.empty(size, dtype="float64")
        #ng = sum(len(c.objs) for c in chunks)
        #mylog.debug("Reading %s cells of %s fields in %s grids",
                    #size, [f2 for f1, f2 in fields], ng)
        #ind = 0
        #for chunk in chunks:
            #data = self._read_chunk_data(chunk, fields)
            #for g in chunk.objs:
                #for field in fields:
                    #ds = data[g.id].pop(field)
                    #nd = g.select(selector, ds, rv[field], ind) # caches
                #ind += nd
                #data.pop(g.id)
        #return rv

    #def _read_chunk_data(self, chunk, fields):
        #data = {}
        #grids_by_file = defaultdict(list)
        #if len(chunk.objs) == 0: return data
        #for g in chunk.objs:
            #if g.filename is None:
                #continue
            #grids_by_file[g.filename].append(g)
        #dtype = self.ds.index._dtype
        #bpr = dtype.itemsize
        #for filename in grids_by_file:
            #grids = grids_by_file[filename]
            #grids.sort(key = lambda a: a._offset)
            #f = open(filename, "rb")
            #for grid in grids:
                #data[grid.id] = {}
                #local_offset = grid._get_offset(f) - f.tell()
                #count = grid.ActiveDimensions.prod()
                #size = count * bpr
                #for field in self.ds.index.field_order:
                    #if field in fields:
                        ## We read it ...
                        #f.seek(local_offset, os.SEEK_CUR)
                        #v = np.fromfile(f, dtype=dtype, count=count)
                        #v = v.reshape(grid.ActiveDimensions, order='F')
                        #data[grid.id][field] = v
                        #local_offset = 0
                    #else:
                        #local_offset += size
        #return data


#class IOHandlerOpalSingle(BaseIOHandler):
    
    #_dataset_type = 'opal_native'
    
    #def __init__(self, ds, *args, **kwargs):
        #print ("IOHandlerOpalSingle")
        #self.ds = ds
    
    