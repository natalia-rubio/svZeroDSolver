import sys
import os
import argparse
import numpy as np
import shutil

from sv_src.solver import Solver0D
from sv_src.misc import m2d, d2m
from sv_src.centerlines import Centerlines

try:
    import sv_rom_extract_results as extract_results
except ImportError:
    print('please run using simvascular --python -- this_script.py')


def project_results(junction_type, model_name):

    solver = Solver0D();
    params = extract_results.Parameters()

    params.data_names = ['flow','pressure']

    params.results_directory = "vmr_test_cases/" + junction_type + "/results_0d/"
    params.solver_file_name = model_name + "_0d.in"
    params.output_directory = "vmr_test_cases/" + junction_type + "/results_cent/"
    params.output_file_name = model_name + "_centerline_results"
    params.centerlines_file = "../../junction_ml/data/centerlines_old/" + model_name + ".vtp"
    params.model_order = 0
    solver.read_solver_file("vmr_test_cases/"+junction_type+ "/input_files/" + params.solver_file_name)
    params.time_range = (solver.inflow.tc * (solver.simulation_params['number_of_cardiac_cycles'] - 1), solver.inflow.tc * solver.simulation_params['number_of_cardiac_cycles'])
    params.oned_model = None
    #ssolver.simulation_params(['number_of_cardiac_cycles'])

    # process
    post = extract_results.Post(params, None)

    print('Writing each timestep...', end = '\t', flush = True)
    post.process()
    print('Done')
    return

#project_results("normal", "0063_1001")

def project_results_all(junction_type):

    with open("vmr_test_cases/" +junction_type+"/input_files/filelist.txt") as f:
            content = f.readlines() # load in models from repository
    models = [x.strip() for x in content].copy()

    for model in models:
        model_name = model[:-6]
        if os.path.exists("vmr_test_cases/" +junction_type+ "/results_cent/"+model_name +"_centerline_results.vtp"):
            continue

        print("\nNow projecting " + model_name + " results onto centerlines.")

        try:
            project_results(junction_type = junction_type, model_name = model_name)
            print("Projection successfull!")
        except:
            print("Error in centerline projection.  Skipping model.")
            import pdb; pdb.set_trace()
            continue

project_results_all(junction_type = "tp_0d")
