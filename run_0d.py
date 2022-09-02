import svzerodsolver
import os
#svzerodsolver.solver.set_up_and_run_0d_simulation('test_cases/git_test_cases/steadyFlow_R_steadyPressure.json')
#svzerodsolver.solver.set_up_and_run_0d_simulation('test_cases/git_test_cases/steadyFlow_TP_junction.json')
#svzerodsolver.solver.set_up_and_run_0d_simulation('tests/cases/steadyFlow_bifurcationR_R1_unified0d.json')
#svzerodsolver.solver.set_up_and_run_0d_simulation('vmr_test_cases/tp_0d_angles/0002_0001_0d.in')
#svzerodsolver.solver.set_up_and_run_0d_simulation('vmr_test_cases/normal_0d_angles/0002_0001_0d.in', use_steady_soltns_as_ics = True)
#svzerodsolver.solver.set_up_and_run_0d_simulation('vmr_test_cases/tp_0d_angles/0063_1001_0d.in')
#svzerodsolver.solver.set_up_and_run_0d_simulation('vmr_test_cases/normal_0d/input_files/0063_1001_0d.in')
def run_all_0d(junction_type):

    with open(f"vmr_test_cases/{junction_type}/input_files/filelist.txt") as f:
            content = f.readlines() # load in models from repository
    models = [x.strip() for x in content].copy()
    models = ["0063_1001_0d.in"]

    for model in models:
        if os.path.exists(f'vmr_test_cases/{junction_type}/results_0d/{model[:-3]}_branch_results.npy'):
            continue
        svzerodsolver.solver.set_up_and_run_0d_simulation(f'vmr_test_cases/{junction_type}/input_files/{model}')
        print(f"Now running model {model}.")
        # try:
        #     svzerodsolver.solver.set_up_and_run_0d_simulation(f'vmr_test_cases/{junction_type}/input_files/{model}')
        # except:
        #     print("Error in 0d simulation.  Skipping model.")
        #     continue

#run_all_0d(junction_type = "normal_0d")
run_all_0d(junction_type = "unified0d_0d")
