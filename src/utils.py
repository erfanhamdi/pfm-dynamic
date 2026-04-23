import os
import numpy as np
import matplotlib.pyplot as plt

def plotter_func(u=None, dim=2, mesh=None, title="", colorbar=False):
    """Save an off-screen PNG of a dolfinx Function.  Works in serial and MPI."""
    from dolfinx import plot  # pylint: disable=import-error

    V = u.ufl_function_space()
    if mesh is None:
        mesh = V.mesh

    comm = mesh.comm
    root = 0
    topology, cell_types, geometry = plot.vtk_mesh(mesh)
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    num_dofs_local  = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    ndpc            = topology[0]   # nodes per cell in VTK connectivity array

    if comm.size > 1:
        # MPI: remap DOF indices to global, gather to rank 0.
        topology_dofs = (np.arange(len(topology)) % (ndpc + 1)) != 0
        topology[topology_dofs] = V.dofmap.index_map.local_to_global(
            topology[topology_dofs].copy()
        )
        g_top  = comm.gather(topology[:(ndpc + 1) * num_cells_local], root=root)
        g_geom = comm.gather(geometry[:V.dofmap.index_map.size_local, :], root=root)
        g_ct   = comm.gather(cell_types[:num_cells_local], root=root)
        g_vals = comm.gather(u.x.array[:num_dofs_local], root=root)
        if comm.rank != root:
            return
        root_top  = np.concatenate(g_top)
        root_geom = np.vstack(g_geom)
        root_ct   = np.concatenate(g_ct)
        root_vals = np.concatenate(g_vals)
    else:
        # Serial: use local arrays directly — no MPI overhead.
        root_top  = topology
        root_geom = geometry
        root_ct   = cell_types
        root_vals = u.x.array[:num_dofs_local]

    import pyvista as pv  # type: ignore  (only imported on rank 0)
    pv.OFF_SCREEN = True
    grid = pv.UnstructuredGrid(root_top, root_ct, root_geom)

    # DG-0 gives one value per cell; Lagrange gives one value per node.
    if len(root_vals) == grid.n_cells:
        grid.cell_data["u"] = root_vals
    else:
        grid.point_data["u"] = root_vals
    grid.set_active_scalars("u")

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid)
    plotter.view_xy()
    plotter.camera.tight()
    if colorbar:
        plotter.add_scalar_bar()
    else:
        plotter.remove_scalar_bar()
    fname = f"{title}.png" if title else "plotter_output.png"
    plotter.screenshot(fname)
    plotter.close()

def plot_force_disp(B, name, out_file):
    plt.figure()
    B_ = np.array(B)
    plt.plot(B_[:, 1], np.abs(B_[:, 0]*1e-3))
    plt.savefig(f"{out_file}/force_disp_{name}.png")
    np.savetxt(f"{out_file}/force_{name}.txt", B_)
    plt.close()

def distance_points_to_segment(points, x1, y1, x2, y2):
    points = np.array(points)
    AB = np.array([x2 - x1, y2 - y1])
    AB_AB = np.dot(AB, AB)
    distances = []
    for point in points:
        px, py = point
        AP = np.array([px - x1, py - y1])
        AP_AB = np.dot(AP, AB)
        t = AP_AB / AB_AB
        if t < 0:
            closest_point = np.array([x1, y1])
        elif t > 1:
            closest_point = np.array([x2, y2])
        else:
            closest_point = np.array([x1, y1]) + t * AB
        
        distance = np.linalg.norm(np.array([px, py]) - closest_point)
        distances.append(distance)
    return np.array(distances)