import vtk
import h5py
import glob
import pyvista as pv
import numpy as np
import meshio
import time
import os
import argparse

# get arg from command line
parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=801)
parser.add_argument("--base_address", type=str, default="/projectnb/lejlab2/erfan/pfmv2/pfm_lite/pfm_dataset/results/tension_amor_1c")
parser.add_argument("--task_id", type=int, default=0)
parser.add_argument("--num_tasks", type=int, default=1)
args = parser.parse_args()
res = args.res
base_address_ = args.base_address
cracks_no = args.base_address.split("/")[-1].split("_")[-1]
initial_cracks_address = f"/projectnb/lejlab2/erfan/pfmv2/pfm_lite/pfm_dataset/initial_cracks_{cracks_no}"
task_idx = 0
sample_size = res
xrng = np.linspace(0, 2, sample_size)
yrng = np.linspace(0, 2, sample_size)
zrng = 0.0
x, y, z= np.meshgrid(xrng, yrng, zrng)
grid = pv.StructuredGrid(x, y, z)
base_address_list = glob.glob(base_address_)
base_address = base_address_list[task_idx]
case_name_list = base_address.split("/")[-1].split("_")[:2]
case_name = "_".join(case_name_list)
print(f"case_name: {case_name}")
case_ = case_name.split("_")[0]
decomposition = case_name.split("_")[1]
if decomposition == 'miehe':
    decomposition = 'spect'
elif decomposition == 'amor':
    decomposition = 'vol'
elif decomposition == 'star':
    decomposition = 'star'
sim_cases = glob.glob(f"{base_address}/*")
sim_cases_with_p_unit = glob.glob(f"{base_address}/*/p_unit.xdmf")
# sim_cases_with_p_unit = ["/projectnb/lejlab2/erfan/pfm_ds/results/phi/shear_star_p_2/364841/p_unit.xdmf"]
# other_sim_cases = [sim_case for sim_case in sim_cases if f"{sim_case}/p_unit.xdmf" not in sim_cases_with_p_unit]
other_sim_cases = []
out_dir = f"/projectnb/lejlab2/erfan/pfmv2/pfm_lite/pfm_dataset/compressed/dataverse/{res}/{case_}/{cracks_no}"
# create the out_dir if it doesn't exist
os.makedirs(out_dir, exist_ok=True)
print(f"compressed file in: {out_dir}")
# time each compression
chunk_size = len(sim_cases_with_p_unit) // args.num_tasks + 1
start = args.task_id * chunk_size
end = min(start + chunk_size, len(sim_cases_with_p_unit))
my_cases = sim_cases_with_p_unit[start:end]
for idx, sim_case in enumerate(my_cases):
    start_time = time.time()
    sim_case = sim_case[:-12]
    sim_case_number = sim_case.split("/")[-1]
    pattern_init = np.load(f"{initial_cracks_address}/{sim_case_number}.npy")
    g = h5py.File(f"{out_dir}/{sim_case_number}.hdf5", 'w')
    print(f"sim case: {sim_case}", flush=True)
    try:
        g_list = []
        p_list = []
        ux_list = []
        uy_list = []
        t_list = []
        sim_case_name = f"{int(sim_case.split("/")[-1]):08d}"
        time_step_idx_list = []
        p_file_add = f"{sim_case}/p_unit.xdmf"
        u_file_add = f"{sim_case}/u_unit.xdmf"
        with meshio.xdmf.TimeSeriesReader(p_file_add) as reader:
            points, cells = reader.read_points_cells()
            for time_step in range(reader.num_steps):
                t, point_data, _ = reader.read_data(time_step)
                mesh = meshio.Mesh(points, cells, point_data=point_data)
                mesh = pv.from_meshio(mesh)
                p_sampled = grid.sample(mesh)
                dst_p = np.array(p_sampled.point_data['f']).reshape([sample_size, sample_size])
                p_list.append(dst_p)
        with meshio.xdmf.TimeSeriesReader(u_file_add) as reader:
            points, cells = reader.read_points_cells()
            for time_step in range(reader.num_steps):
                t, point_data, _ = reader.read_data(time_step)
                t_list.append(t)
                mesh = meshio.Mesh(points, cells, point_data=point_data)
                mesh = pv.from_meshio(mesh)
                u_sampled = grid.sample(mesh)
                dst_u = np.array(u_sampled.point_data['f'])
                dst_ux = dst_u[:, 0].reshape([sample_size, sample_size])
                dst_uy = dst_u[:, 1].reshape([sample_size, sample_size])
                ux_list.append(dst_ux)
                uy_list.append(dst_uy)
        # concatenate the p and u arrays along a third axis
        pu_array = np.stack((np.array(p_list), np.array(ux_list), np.array(uy_list)))
        if case_ == "tension":
            forces_y = np.loadtxt(f"{sim_case}/force_bot_rxn.txt")
            forces_x = np.loadtxt(f"{sim_case}/force_left_rxn.txt")
            g.create_dataset(f"{sim_case_name}/force_disp_y", data=forces_y, compression="gzip", compression_opts=9, shuffle=True, chunks=True)
            g.create_dataset(f"{sim_case_name}/force_disp_x", data=forces_x, compression="gzip", compression_opts=9, shuffle=True, chunks=True)
        else:
            forces_x = np.loadtxt(f"{sim_case}/force_bot_rxn.txt")
            g.create_dataset(f"{sim_case_name}/force_disp_x", data=forces_x, compression="gzip", compression_opts=9, shuffle=True, chunks=True)
    
        g.create_dataset(f"{sim_case_name}/init", data=pattern_init, compression="gzip", compression_opts=9, shuffle=True, chunks=True)
        print(pu_array.shape)
        g.create_dataset(f"{sim_case_name}/data", data=pu_array, compression="gzip", compression_opts=9, shuffle=True, chunks=True)
        g.create_dataset(f"{sim_case_name}/grid/x", data=grid.points[:, 0], compression="gzip", compression_opts=9, shuffle=True, chunks=True)
        g.create_dataset(f"{sim_case_name}/grid/y", data=grid.points[:, 1], compression="gzip", compression_opts=9, shuffle=True, chunks=True)
        time_step_idx_array = np.array(t_list)
        g.create_dataset(f"{sim_case_name}/grid/t", data=time_step_idx_array, compression="gzip", compression_opts=9, shuffle=True, chunks=True)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
    except Exception as e:
        print(e)
        continue
    g.close()