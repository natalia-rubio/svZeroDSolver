import argparse
import glob
import os
import csv
import re
import sys
import scipy
import pdb
import vtk
from scipy.interpolate import interp1d

import numpy as np
from collections import defaultdict, OrderedDict

from common import get_dict
from get_database import input_args
from vtk_functions import read_geo, write_geo, collect_arrays, get_all_arrays, ClosestPoints
from get_bc_integrals import get_res_names
from vtk_to_xdmf import write_xdmf

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

def map_rom_to_centerline(rom, geo_cent, res, time, only_last=True):
    """
    ROM: 
    Map 0d or 1d results to centerline
    """
    # assemble output dict
    rec_dd = lambda: defaultdict(rec_dd)
    arrays = rec_dd()

    # get centerline arrays
    arrays_cent, _ = get_all_arrays(geo_cent)

    # centerline points
    points = v2n(geo_cent.GetPoints().GetData())

    # pick results
    if only_last:
        name = rom + '_int_last'
        t_vec = time[rom]
    else:
        name = rom + '_int'
        t_vec = time[rom + '_all']

    # loop all result fields
    for f in res[0].keys():
        if 'path' in f:
            continue
        array_f = np.zeros((arrays_cent['Path'].shape[0], len(t_vec)))
        n_outlet = np.zeros(arrays_cent['Path'].shape[0])
        for br in res.keys():
            # get centerline path
            path_cent = arrays_cent['Path'][arrays_cent['BranchId'] == br]
            path_cent /= path_cent[-1]

            # get 0d path
            path_0d = res[br][rom + '_path']
            path_0d /= path_0d[-1]

            # linearly interpolate results along centerline
            f_cent = interp1d(path_0d, res[br][f][name].T)(path_cent).T

            # store in global array
            array_f[arrays_cent['BranchId'] == br] = f_cent

            # add upstream part of branch within junction
            if br == 0:
                continue

            # first point of branch
            ip = np.where(arrays_cent['BranchId'] == br)[0][0]

            # centerline that passes through branch (first occurence)
            cid = np.where(arrays_cent['CenterlineId'][ip])[0][0]

            # id of upstream junction
            jc = arrays_cent['BifurcationId'][ip - 1]

            # centerline within junction
            jc_cent = np.where(np.logical_and(arrays_cent['BifurcationId'] == jc,
                                              arrays_cent['CenterlineId'][:, cid]))[0]

            # length of centerline within junction
            jc_path = np.append(0, np.cumsum(np.linalg.norm(np.diff(points[jc_cent], axis=0), axis=1)))
            jc_path /= jc_path[-1]

            # results at upstream branch
            res_br_u = res[arrays_cent['BranchId'][jc_cent[0] - 1]][f][name]

            # results at beginning and end of centerline within junction
            f0 = res_br_u[-1]
            f1 = res[br][f][name][0]

            # map 1d results to centerline using paths
            array_f[jc_cent] += interp1d([0, 1], np.vstack((f0, f1)).T, fill_value='extrapolate')(jc_path).T

            # count number of outlets of this junction
            n_outlet[jc_cent] += 1

            # normalize by number of outlets
        array_f[n_outlet > 0] = (array_f[n_outlet > 0].T / n_outlet[n_outlet > 0]).T

        # assemble time steps
        if only_last:
            arrays[0]['point'][f] = array_f[:, -1]
        else:
            for i, t in enumerate(t_vec):
                arrays[str(t)]['point'][f] = array_f[:, i]

    return arrays
