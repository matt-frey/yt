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
