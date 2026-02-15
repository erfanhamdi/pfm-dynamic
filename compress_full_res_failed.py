import vtk
import vtk
import h5py
import glob
import pyvista as pv
import numpy as np
import meshio
import matplotlib.pyplot as plt
import os
import argparse

# get arg from command line
parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=801)
parser.add_argument("--base_address", type=str, default="/projectnb/lejlab2/erfan/pfmv2/pfm_lite/pfm_dataset/results/tension_amor_1c")
parser.add_argument("--failed_cases", type=str, default="/projectnb/lejlab2/erfan/pfmv2/pfm_lite/pfm_dataset/remaining_cases.txt")
args = parser.parse_args()
res = args.res
base_address_ = args.base_address
failed_cases = args.failed_cases
with open(failed_cases, 'r') as f:
    failed_cases = f.readlines()
failed_cases = [case.strip() for case in failed_cases]
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
sim_cases_with_p_unit = [case for case in sim_cases if case in failed_cases]
# sim_cases_with_p_unit = ["/projectnb/lejlab2/erfan/pfm_ds/results/phi/shear_star_p_2/364841/p_unit.xdmf"]
# other_sim_cases = [sim_case for sim_case in sim_cases if f"{sim_case}/p_unit.xdmf" not in sim_cases_with_p_unit]
other_sim_cases = []
out_dir = f"/projectnb/lejlab2/erfan/pfmv2/pfm_lite/pfm_dataset/compressed/dataverse/{res}/{case_}/{cracks_no}"
# create the out_dir if it doesn't exist
os.makedirs(out_dir, exist_ok=True)
print(f"compressed file in: {out_dir}")
# time each compression
import time
for idx, sim_case in enumerate(sim_cases_with_p_unit):
    start_time = time.time()
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
for idx, sim_case in enumerate(other_sim_cases):
    start_time = time.time()
    print(f"sim case: {sim_case}")
    sim_case_number = sim_case.split("/")[-1]
    pattern_init = np.load(f"/projectnb/lejlab2/erfan/pfm_ds/initial_cracks/{sim_case_number}.npy")

    g = h5py.File(f"{out_dir}/{sim_case_number}.hdf5", 'w')
    try:
        g_list = []
        p_list = []
        ux_list = []
        uy_list = []
        # sim_case_name = f"{int(sim_case.split("/")[-1]):06d}"
        sim_case_name = f"{int(sim_case.split("/")[-1]):08d}"
        out_files_p = glob.glob(f"{sim_case}/p*.h5")
        out_files_u = glob.glob(f"{sim_case}/u*.h5")
        time_step_idx_list = []
        if len(out_files_p) == 101:
            for out_file in out_files_p:
                time_step_idx_list.append(out_file.split("/")[-1].split("_")[-1].split(".")[0])
            # sort the time step indices
            time_step_idx_list = sorted(time_step_idx_list, key=lambda x: int(x))
            
            for idx, time_step in enumerate(time_step_idx_list):
                # p = h5py.File(f"{sim_case}/p_mpi_{time_step}.h5", 'r')
                p_file_add = f"{sim_case}/p_{time_step}.xdmf"
                u_file_add = f"{sim_case}/u_{time_step}.xdmf"
                # u = h5py.File(f"{sim_case}/u_mpi_{time_step}.h5", 'r')
                with meshio.xdmf.TimeSeriesReader(p_file_add) as reader:
                    points, cells = reader.read_points_cells()
                    t, point_data, _ = reader.read_data(0)
                    mesh = meshio.Mesh(points, cells, point_data=point_data)
                    mesh = pv.from_meshio(mesh)
                p_sampled = grid.sample(mesh)
                with meshio.xdmf.TimeSeriesReader(u_file_add) as reader:
                    points, cells = reader.read_points_cells()
                    t, point_data, _ = reader.read_data(0)
                    mesh = meshio.Mesh(points, cells, point_data=point_data)
                    mesh = pv.from_meshio(mesh)
                u_sampled = grid.sample(mesh)
                dst_p = np.array(p_sampled.point_data['f']).reshape([sample_size, sample_size])
                dst_u = np.array(u_sampled.point_data['f'])
                dst_ux = dst_u[:, 0].reshape([sample_size, sample_size])
                dst_uy = dst_u[:, 1].reshape([sample_size, sample_size])
                # concatenate the p and u arrays along a third axis
                pu_array = np.stack((dst_p, dst_ux, dst_uy))
                g_list.append(pu_array)
        # pu_array = np.stack((np.array(p_list), np.array(ux_list), np.array(uy_list)))
            if case_ == "tension":
                forces_y = np.loadtxt(f"{sim_case}/force_bot_rxn.txt")
                forces_x = np.loadtxt(f"{sim_case}/force_left_rxn.txt")
                g.create_dataset(f"{sim_case_name}/force_disp_y", data=forces_y, compression="gzip", compression_opts=9, shuffle=True, chunks=True)
                g.create_dataset(f"{sim_case_name}/force_disp_x", data=forces_x, compression="gzip", compression_opts=9, shuffle=True, chunks=True)
            else:
                forces_x = np.loadtxt(f"{sim_case}/force_bot_rxn.txt")
                g.create_dataset(f"{sim_case_name}/force_disp_x", data=forces_x, compression="gzip", compression_opts=9, shuffle=True, chunks=True)
            g.create_dataset(f"{sim_case_name}/init", data=pattern_init, compression="gzip", compression_opts=9, shuffle=True, chunks=True)
            g_list = np.transpose(np.array(g_list), (1, 0, 2, 3))
            print(g_list.shape)
            g.create_dataset(f"{sim_case_name}/data", data=g_list, compression="gzip", compression_opts=9, shuffle=True, chunks=True)
            g.create_dataset(f"{sim_case_name}/grid/x", data=grid.points[:, 0], compression="gzip", compression_opts=9, shuffle=True, chunks=True)
            g.create_dataset(f"{sim_case_name}/grid/y", data=grid.points[:, 1], compression="gzip", compression_opts=9, shuffle=True, chunks=True)
            # time_step_idx_array = np.array(t_list)
            g.create_dataset(f"{sim_case_name}/grid/t", data=time_step_idx_array, compression="gzip", compression_opts=9, shuffle=True, chunks=True)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds", flush=True)
    except Exception as e:
        print(e)
        continue
    g.close()

# g.close()
# with h5py.File("/projectnb/lejlab2/erfan/dataset_phase1/data/b4/compressed.hdf5", 'r') as h5_file:
#             seed_group = h5_file['000705']
#             data = np.array(seed_group["data"], dtype='f')