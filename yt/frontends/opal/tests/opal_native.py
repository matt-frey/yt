"""
OPAL native frontend tests



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2016, PSI.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import yt

from yt.testing import \
    assert_equal, \
    requires_file, \
    units_override_check
from yt.utilities.answer_testing.framework import \
    requires_ds, \
    small_patch_amr, \
    data_dir_load
from yt.frontends.opal.api import OpalSingleDataset

ds = yt.load("RingCyclotronMatched.h5")
    
#ds.print_stats()
    
#print ("Field list:", ds.field_list)
#print ("Derived field list:", ds.derived_field_list)    