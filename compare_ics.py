import numpy as np
import pickle
import pandas as pd

normal_dir = "vmr_test_cases/normal_0d_angles"
total_pressure_dir = "vmr_test_cases/tp_0d_angles"

#model_name = "0063_1001"
model_name = "0002_0001"

normal_ics = np.load(f"{normal_dir}/{model_name}_0d_initial_conditions.npy", allow_pickle = True).item()
total_pressure_ics = np.load(f"{total_pressure_dir}/{model_name}_0d_initial_conditions.npy", allow_pickle = True).item()
import pdb; pdb.set_trace()
ic_data = {"Variable Name": normal_ics["var_name_list"],
        "Y (Normal)": list(normal_ics["y"]),
        "Y (Total Pressure)": list(total_pressure_ics["y"]),
        "Y (Relative Error)": list(np.abs((normal_ics["y"]-total_pressure_ics["y"])/normal_ics["y"])),
        "Ydot (Normal)": list(normal_ics["ydot"]),
        "Ydot (Total Pressure)": list(total_pressure_ics["ydot"]),
        "Ydot (Relative Error)": list(np.abs((normal_ics["ydot"]-total_pressure_ics["ydot"])/normal_ics["ydot"]))
        }
df = pd.DataFrame(ic_data)
df.to_csv(f"{model_name}_ic_comparison.csv")
