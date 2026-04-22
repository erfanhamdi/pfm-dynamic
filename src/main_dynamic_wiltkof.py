import ufl
import numpy as np
import glob
import time
import os

from dolfinx import mesh, fem, io, default_scalar_type, log
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector, create_matrix
from mpi4py import MPI
from petsc4py import PETSc
from pathlib import Path

from utils import plotter_func, plot_force_disp, distance_points_to_segment
import argparse

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["HDF5_MPI_OPS_COLLECTIVE"] = "TRUE"

parser = argparse.ArgumentParser(description='2D internal cracks - DYNAMIC tension')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--mesh_size', type=int, default=400)
parser.add_argument('--prefix', type=str, default="test")
parser.add_argument('--job_id', type=int, default=0)
parser.add_argument('--g_c', type=float, default=2.213e4)
parser.add_argument('--e_', type=float, default=190.0e9)
# ── CHANGE 1 ──────────────────────────────────────────────────────────────────
# rho: material density — new parameter required for inertia term.
# Units must be consistent with E (Pa) and length (m).  E.g. steel: rho=8000 kg/m³,
# but if your E is in kPa scale, adjust accordingly.
parser.add_argument('--rho', type=float, default=8000.0, help='Mass density kg/m3')
# ── END CHANGE 1 ──────────────────────────────────────────────────────────────
parser.add_argument('--domain_size', type=float, default=2.0)
parser.add_argument('--num_steps', type=int, default=5000)
parser.add_argument('--init_crack_add', type=str, default="pfm_lite/initial_cracks")
# ── CHANGE 2 ──────────────────────────────────────────────────────────────────
# delta_T is now the PHYSICAL time step (seconds), NOT a loading ramp parameter.
# It will be computed automatically from the CFL condition below; this argument
# becomes a hard upper cap (or set it to 0 to let CFL determine everything).
parser.add_argument('--delta_T', type=float, default=0.0, help='Override dt (0 = use CFL)')
# ── END CHANGE 2 ──────────────────────────────────────────────────────────────
# ── CHANGE 3 ──────────────────────────────────────────────────────────────────
# Total physical simulation time (seconds).  Replace the old "num_steps * delta_T"
# loading ramp with an actual time horizon.
parser.add_argument('--T_final', type=float, default=90e-5, help='Total physical time (s)')
# ── END CHANGE 3 ──────────────────────────────────────────────────────────────
# ── CHANGE 4 ──────────────────────────────────────────────────────────────────
# Loading ramp parameters: the imposed velocity is ramped up linearly over T_ramp
# seconds, then held constant at v_imp (m/s).  In quasi-static you used
# delta_T * step as the ramp; here we use a smooth velocity ramp so the boundary
# condition has a well-defined velocity AND acceleration (needed for the explicit
# time integrator to initialise cleanly).
parser.add_argument('--v_imp', type=float, default=16.5, help='Imposed velocity (m/s)')
parser.add_argument('--T_ramp', type=float, default=1e-6, help='Velocity ramp-up time (s)')
parser.add_argument('--l_0', type=float, default=0.0,
                    help='Phase-field length scale (m); 0 = auto (2*h)')
parser.add_argument('--nu', type=float, default=0.3, help='Poisson ratio')
# ── END CHANGE 4 ──────────────────────────────────────────────────────────────
args = parser.parse_args()

seed = args.seed - 1
mesh_size = args.mesh_size
prefix = args.prefix
job_id = args.job_id
g_c_ = args.g_c
e_ = args.e_
rho_ = args.rho          # NEW
domain_size = args.domain_size
T_final = args.T_final
v_imp = args.v_imp
T_ramp = args.T_ramp

# Tension case only — no branch needed
sim_case = "tension"

start_time = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Running DYNAMIC tension on {size} MPI processes")
    print(f"Job id: {job_id}")

ksp = PETSc.KSP.Type.GMRES
pc = PETSc.PC.Type.HYPRE
rtol = 1e-8
max_it = 1000

initial_cracks = glob.glob(f"{args.init_crack_add}/*.npy")
initial_cracks = sorted(initial_cracks, key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
crack_pattern = np.load(initial_cracks[seed])
seed_val = int(initial_cracks[seed].split("/")[-1].split(".")[0])
out_file = f"pfm_lite/results/{prefix}/{seed_val}"
results_folder = Path(out_file)

domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([domain_size, domain_size])],
    [mesh_size, mesh_size],
    cell_type=mesh.CellType.quadrilateral
)

if rank == 0:
    print(f"seed: {seed_val}, out_file = {out_file}", flush=True)
    Path(out_file).mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(exist_ok=True, parents=True)

try:
    out_file_name   = io.XDMFFile(domain.comm, f"{out_file}/p_unit.xdmf", 'w')
    out_file_name.write_mesh(domain)
    out_file_name_u = io.XDMFFile(domain.comm, f"{out_file}/u_unit.xdmf", 'w')
    out_file_name_u.write_mesh(domain)
    file_init_success = True
except RuntimeError as e:
    if rank == 0:
        print(f"Warning: Could not initialise output files: {str(e)}")
    file_init_success = False

# ── Material constants ──────────────────────────────────────────────────────
nu_  = float(args.nu)
G_c_ = fem.Constant(domain, g_c_)
E    = fem.Constant(domain, e_)
nu   = fem.Constant(domain, nu_)
mu     = E / (2 * (1 + nu))
lmbda  = E * nu / ((1 + nu) * (1 - 2 * nu))
n      = fem.Constant(domain, 3.0)
Kn     = lmbda + 2 * mu / n
rho    = fem.Constant(domain, rho_)

# Mesh size and CFL-stable time step.
#   c_p = sqrt(E*(1-nu) / (rho*(1+nu)*(1-2*nu)))   (P-wave speed)
# safety_factor is a divisor: dt = h_min / c_p / safety_factor.
safety_factor = 5.0
h_min = domain.comm.allreduce(
    np.min(domain.h(domain.topology.dim, np.arange(domain.topology.index_map(domain.topology.dim).size_local))),
    op=MPI.MIN
)

# Phase-field length scale: require l_0 >= 2*h for mesh-independent results.
if args.l_0 > 0.0:
    l_0_val = args.l_0
else:
    l_0_val = 2.0 * h_min
l_0_ = fem.Constant(domain, l_0_val)

c_p_val = float(np.sqrt(e_ * (1.0 - nu_) / (rho_ * (1.0 + nu_) * (1.0 - 2.0 * nu_))))
dt_cfl  = h_min / c_p_val / safety_factor

if args.delta_T > 0.0:
    dt_val = min(args.delta_T, dt_cfl)
else:
    dt_val = dt_cfl

dt = fem.Constant(domain, dt_val)   # physical time step (s)

if rank == 0:
    print(f"h_min = {h_min:.4e} m,  c_p = {c_p_val:.2f} m/s,  dt_CFL = {dt_cfl:.4e} s,  dt_used = {dt_val:.4e} s")
    print(f"l_0 = {l_0_val:.4e} m  (h_min = {h_min:.4e} m,  ratio l_0/h = {l_0_val/h_min:.2f})")

num_steps = int(np.ceil(T_final / dt_val))
save_freq  = max(1, num_steps // 200)

if rank == 0:
    print(f"num_steps = {num_steps},  save_freq = {save_freq}")

# ── Function spaces ─────────────────────────────────────────────────────────
V  = fem.functionspace(domain, ("Lagrange", 1,))
W  = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
VV = fem.functionspace(domain, ("DG", 0,))

# Trial/test functions — used for the phase-field bilinear form only
p, q = ufl.TrialFunction(V), ufl.TestFunction(V)

# ── CHANGE 7 ──────────────────────────────────────────────────────────────────
# Kinematic fields.  We need THREE displacement-like fields:
#   u_n     : displacement at t_n   (start of step)
#   v_n     : velocity   at t_n
#   a_n     : acceleration at t_n
#   u_new   : displacement at t_{n+1} (predicted, then corrected)
#   v_new   : velocity   at t_{n+1}
#   a_new   : acceleration at t_{n+1}
# In the old quasi-static code only u_new and u_old existed.
u_n   = fem.Function(W, name="u_n")    # displacement at t_n
v_n   = fem.Function(W, name="v_n")    # velocity     at t_n
a_n   = fem.Function(W, name="a_n")    # acceleration at t_n
u_new = fem.Function(W, name="u_new")  # displacement at t_{n+1}
v_new = fem.Function(W, name="v_new")  # velocity     at t_{n+1}
a_new = fem.Function(W, name="a_new")  # acceleration at t_{n+1}
# ── END CHANGE 7 ──────────────────────────────────────────────────────────────

p_new, H_old, p_old = fem.Function(V), fem.Function(VV), fem.Function(V)
H_init_ = fem.Function(V)

tdim = domain.topology.dim
fdim = tdim - 1

# ── Boundary markers ─────────────────────────────────────────────────────────
def top_boundary(x):    return np.isclose(x[1], domain_size)
def left_boundary(x):   return np.isclose(x[0], 0)
def right_boundary(x):  return np.isclose(x[0], domain_size)
def bottom_boundary(x): return np.isclose(x[1], 0.0)

top_facet   = mesh.locate_entities_boundary(domain, fdim, top_boundary)
bot_facet   = mesh.locate_entities_boundary(domain, fdim, bottom_boundary)
right_facet = mesh.locate_entities_boundary(domain, fdim, right_boundary)
left_facet  = mesh.locate_entities_boundary(domain, fdim, left_boundary)

marked_facets = np.hstack([top_facet, bot_facet, right_facet, left_facet])
marked_values = np.hstack([
    np.full_like(top_facet, 1),
    np.full_like(bot_facet, 2),
    np.full_like(right_facet, 3),
    np.full_like(left_facet, 4)
])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

top_y_dofs  = fem.locate_dofs_topological(W.sub(1), fdim, top_facet)
bot_y_dofs  = fem.locate_dofs_topological(W.sub(1), fdim, bot_facet)
bot_x_dofs  = fem.locate_dofs_topological(W.sub(0), fdim, bot_facet)
right_x_dofs = fem.locate_dofs_topological(W.sub(0), fdim, right_facet)
left_x_dofs  = fem.locate_dofs_topological(W.sub(0), fdim, left_facet)
left_y_dofs  = fem.locate_dofs_topological(W.sub(1), fdim, left_facet)
top_phi_dofs = fem.locate_dofs_topological(V, fdim, top_facet)
# ── CHANGE 8 ──────────────────────────────────────────────────────────────────
# Boundary conditions for the EXPLICIT dynamic scheme.
#
# Quasi-static:  u_bc_top.value = delta_T * step   (displacement ramp)
# Dynamic:       we prescribe DISPLACEMENT, VELOCITY and ACCELERATION
#                separately, because the Verlet predictor sets u_new from
#                u_n/v_n/a_n, and we must enforce the Dirichlet BC on
#                u_new AFTER the predictor (see time loop below).
#
# We keep the same DOFs as before (top y, bottom fixed, left/right lateral).
# The time-varying values are updated every step as:
#
#   t       current physical time
#   u_imp   displacement  = v_imp * (t²/(2*T_ramp))  for t <= T_ramp
#                         = v_imp * (t - T_ramp/2)     for t >  T_ramp
#   v_imp_t velocity      = v_imp * t/T_ramp           for t <= T_ramp
#                         = v_imp                       for t >  T_ramp
#   a_imp_t acceleration  = v_imp / T_ramp             for t <= T_ramp
#                         = 0                           for t >  T_ramp
#
# These are stored as FEniCSx Constants and updated in the loop.
u_bc_top_disp = fem.Constant(domain, default_scalar_type(0.0))  # u_imp(t)
u_bc_top_vel  = fem.Constant(domain, default_scalar_type(0.0))  # v_imp(t)
u_bc_top_acc  = fem.Constant(domain, default_scalar_type(0.0))  # a_imp(t)

# Displacement BCs (applied to u_new after Verlet predictor)
bc_bot_y      = fem.dirichletbc(default_scalar_type(0.0), bot_y_dofs,   W.sub(1))
bc_bot_x      = fem.dirichletbc(default_scalar_type(0.0), bot_x_dofs,   W.sub(0))
bc_top_y_disp = fem.dirichletbc(u_bc_top_disp, top_y_dofs, W.sub(1))
bc_left_x     = fem.dirichletbc(default_scalar_type(0.0), left_x_dofs,  W.sub(0))
bc_right_x    = fem.dirichletbc(default_scalar_type(0.0), right_x_dofs, W.sub(0))
bc_phi_top = fem.dirichletbc(default_scalar_type(0.0), top_phi_dofs, V)

bc_u = [bc_bot_y, bc_top_y_disp, bc_bot_x]
bc_phi = []

# Velocity BCs (applied to v_new after velocity corrector)
bc_top_y_vel = fem.dirichletbc(u_bc_top_vel, top_y_dofs, W.sub(1))
bc_v = [bc_bot_y, bc_top_y_vel]

# Acceleration BCs (applied to a_new after acceleration update)
bc_top_y_acc = fem.dirichletbc(u_bc_top_acc, top_y_dofs, W.sub(1))
bc_a = [bc_bot_y, bc_top_y_acc]
# ── END CHANGE 8 ──────────────────────────────────────────────────────────────

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})

# ── Kinematics helpers ───────────────────────────────────────────────────────
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(2) + 2.0 * mu * epsilon(u)

def bracket_pos(u):
    return 0.5 * (u + abs(u))

def bracket_neg(u):
    return 0.5 * (u - abs(u))

# ── CHANGE 9 ──────────────────────────────────────────────────────────────────
# Spectral decomposition must now reference u_new (same as before), but note
# that in the dynamic loop u_new is the PREDICTED displacement at t_{n+1}.
# The spectral decomposition and Miehe psi_pos/psi_neg are unchanged in form —
# only the displacement field they act on changes from quasi-static to dynamic.
# ── END CHANGE 9 ──────────────────────────────────────────────────────────────
A = ufl.variable(epsilon(u_new))
I1    = ufl.tr(A)
delta = (A[0, 0] - A[1, 1])**2 + 4 * A[0, 1] * A[1, 0] + 3.0e-16**2
eigval_1 = (I1 - ufl.sqrt(delta)) / 2
eigval_2 = (I1 + ufl.sqrt(delta)) / 2
eigvec_1 = ufl.diff(eigval_1, A).T
eigvec_2 = ufl.diff(eigval_2, A).T
epsilon_p = (0.5 * (eigval_1 + abs(eigval_1)) * eigvec_1
           + 0.5 * (eigval_2 + abs(eigval_2)) * eigvec_2)
epsilon_n = (0.5 * (eigval_1 - abs(eigval_1)) * eigvec_1
           + 0.5 * (eigval_2 - abs(eigval_2)) * eigvec_2)

# Miehe decomposition (tension case only — psi_pos / psi_neg)
def psi_pos_m(u):
    return (0.5 * lmbda * (bracket_pos(ufl.tr(epsilon(u)))**2)
            + mu * ufl.inner(epsilon_p, epsilon_p))

def psi_neg_m(u):
    return (0.5 * lmbda * (bracket_neg(ufl.tr(epsilon(u)))**2)
            + mu * ufl.inner(epsilon_n, epsilon_n))

psi_pos = psi_pos_m(u_new)
psi_neg = psi_neg_m(u_new)

def H(u_new, H_old):
    return ufl.conditional(ufl.gt(psi_pos, H_old), psi_pos, H_old)

# ── Initial crack field ──────────────────────────────────────────────────────
def H_init(dist_list, l_0, G_c):
    distances = np.array(dist_list)
    distances = np.min(distances, axis=0)
    mask0 = distances <= l_0.value / 2
    H = np.zeros_like(distances)
    phi_c = 0.999
    H[mask0] = ((phi_c / (1 - phi_c)) * G_c.value / (2 * l_0.value)) * (1 - (2 * distances[mask0] / l_0.value))
    return H

A_ = crack_pattern[:, 0, :] / (2 / domain_size)
B_ = crack_pattern[:, 1, :] / (2 / domain_size)
points = domain.geometry.x[:, :2]
dist_list = []
for idx in range(len(A_)):
    distances = distance_points_to_segment(points, A_[idx][0], A_[idx][1], B_[idx][0], B_[idx][1])
    dist_list.append(distances)
H_init_.x.array[:] = H_init(dist_list, l_0_, G_c_)
H_old.interpolate(H_init_)

# ── CHANGE 10 ─────────────────────────────────────────────────────────────────
# Phase-field problem (UNCHANGED from quasi-static).
# The phase-field equation has no time derivative — it is solved as a linear
# system at each step using the current (predicted) u_new.  This is the
# staggered approach: mechanics updates u, then this solves for phi.
# ── END CHANGE 10 ─────────────────────────────────────────────────────────────
T_traction = fem.Constant(domain, default_scalar_type((0, 0)))
E_phi = (
    (l_0_**2) * ufl.dot(ufl.grad(p), ufl.grad(q))
    + ((2 * l_0_ / G_c_) * H(u_new, H_old) + 1) * p * q
) * dx - (2 * l_0_ / G_c_) * H(u_new, H_old) * q * dx

a_phi = fem.form(ufl.lhs(E_phi))
L_phi = fem.form(ufl.rhs(E_phi))
A_phi = create_matrix(a_phi)
b_phi = create_vector(L_phi)

solver_phi = PETSc.KSP().create(domain.comm)
solver_phi.setOperators(A_phi)
solver_phi.setType(ksp)
solver_phi.getPC().setType(pc)
solver_phi.setTolerances(rtol=rtol, max_it=max_it)
solver_phi.setFromOptions()

# ── CHANGE 11 ─────────────────────────────────────────────────────────────────
# LUMPED MASS ASSEMBLY.
#
# Assembled directly on the vector space W so the DOF ordering matches
# F_int_vec exactly.  action(rho*inner(u,v)*dx, ones_W) gives the row sums
# of the consistent mass matrix as a W-sized vector — the lumped diagonal.
# rho is baked in so the acceleration divide is simply  a = F_int / M_lumped.
u_trial_m = ufl.TrialFunction(W)
v_test_m  = ufl.TestFunction(W)
ones_W    = fem.Function(W)
ones_W.x.array[:] = 1.0

M_action_compiled = fem.form(ufl.action(rho * ufl.inner(u_trial_m, v_test_m) * dx, ones_W))
# Back the lumped mass with a Function so .x.array has the same local+ghost
# layout as a_n/v_n/u_n — essential for the element-wise divide below.
M_func = fem.Function(W)
M_vec  = M_func.x.petsc_vec
with M_vec.localForm() as loc:
    loc.set(0)
assemble_vector(M_vec, M_action_compiled)
M_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
M_func.x.scatter_forward()

M_lumped_local = M_func.x.array.copy()

min_mass_local  = float(M_lumped_local.min()) if M_lumped_local.size else np.inf
min_mass_global = domain.comm.allreduce(min_mass_local, op=MPI.MIN)
if min_mass_global <= 0.0:
    raise RuntimeError(
        f"Lumped mass has non-positive entry (min={min_mass_global:.3e}); "
        f"check mesh connectivity and rho."
    )
# ── END CHANGE 11 ─────────────────────────────────────────────────────────────

# ── CHANGE 12 ─────────────────────────────────────────────────────────────────
# Internal force form F_int = -dE_elas/du evaluated at (u_new, p_new).
#
# In the quasi-static code the displacement problem was:
#   E_du = ((1-p_new)**2) * inner(grad(v), sigma(u)) * dx
# which assembled into a global linear system  K * u = F_ext.
#
# In the explicit dynamic scheme we NEVER solve a global system for u.
# Instead we:
#   (a) predict u_new kinematically (Verlet predictor),
#   (b) assemble F_int = integral of degraded stress against test functions,
#   (c) divide by the lumped mass to get acceleration: a_new = F_int / M.
#
# The UFL form for F_int is the residual of the elastic weak form with u=u_new
# and p=p_new fixed.  Note the MINUS sign: a_new = +F_int because the residual
# convention is  M*a + F_int = F_ext, and here F_ext = 0 (no body force).
# Neumann BCs (surface tractions) would be added here as ufl.dot(T, v)*ds.
v_test = ufl.TestFunction(W)
F_int_form = fem.form(
    -((1.0 - p_new)**2) * ufl.inner(ufl.grad(v_test), sigma(u_new)) * dx
)
# Back F_int with a Function so .x.array has the same local+ghost layout
# as a_n.x.array.  F_int_vec is just the underlying PETSc handle for assembly.
F_int_func = fem.Function(W)
F_int_vec  = F_int_func.x.petsc_vec
# ── END CHANGE 12 ─────────────────────────────────────────────────────────────

# ── Reaction force measurement (tension: bottom y and left x) ────────────────
B_bot  = []
B_left = []

v_reac = fem.Function(W)
# Virtual-work reaction: elastic internal force + inertial force (M*a).
# In dynamics the Dirichlet reaction must include rho*a*v, otherwise the
# reported force omits the inertial contribution (significant during ramp-up).
virtual_work_form = fem.form(ufl.action(
    ((1.0 - p_new)**2) * ufl.inner(ufl.grad(v_test), sigma(u_new)) * dx
    + rho * ufl.inner(a_new, v_test) * dx,
    v_reac
))

bot_dofs_geo   = fem.locate_dofs_geometrical(W, bottom_boundary)
u_bc_bot_func  = fem.Function(W)
bc_bot_rxn     = fem.dirichletbc(u_bc_bot_func, bot_dofs_geo)

left_dofs_geo  = fem.locate_dofs_geometrical(W, left_boundary)
u_bc_left_func = fem.Function(W)
bc_left_rxn    = fem.dirichletbc(u_bc_left_func, left_dofs_geo)

def one(x):
    values = np.zeros((1, x.shape[1]))
    values[0] = 1.0
    return values

u_bc_bot_func.sub(1).interpolate(one)
u_bc_left_func.sub(0).interpolate(one)

H_expr = fem.Expression(
    ufl.conditional(ufl.gt(psi_pos, H_old), psi_pos, H_old),
    VV.element.interpolation_points()
)

# ── CHANGE 13 ─────────────────────────────────────────────────────────────────
# MAIN TIME LOOP — explicit Verlet + staggered phase-field.
#
# Each step does:
#   A. Update loading (displacement, velocity, acceleration imposed values)
#   B. Verlet predictor:  u_new = u_n + dt*v_n + 0.5*dt²*a_n
#      Apply Dirichlet BCs on u_new (overwrite predicted DOFs with imposed u)
#   C. Phase-field solve: phi_{n+1} from linear system using u_new
#   D. Assemble F_int(u_new, phi_{n+1})
#   E. Acceleration:  a_new = F_int / (rho * M_lumped)  — element-wise divide
#      Apply Dirichlet BCs on a_new
#   F. Velocity corrector: v_new = v_n + 0.5*dt*(a_n + a_new)
#      Apply Dirichlet BCs on v_new
#   G. Update state:  u_n ← u_new, v_n ← v_new, a_n ← a_new
#      Update history field H_old (irreversibility)
#
# There is NO staggered iteration loop (flag/while) unlike quasi-static.
# The explicit scheme computes each field once per step.
# ── END CHANGE 13 ─────────────────────────────────────────────────────────────

dt_val_f = float(dt.value)

def loading(t):
    """C^2-smooth ramp: 5th-order polynomial on [0, T_ramp], linear after.

    Satisfies u(0)=v(0)=a(0)=0, v(T_ramp)=v_imp, a(T_ramp)=0 so the boundary
    kinematics are continuous with the interior initial conditions (no shock).
    """
    if t <= T_ramp:
        tau = t / T_ramp
        u  = v_imp * T_ramp * tau**4 * (2.5 - 3.0 * tau + tau**2)
        v  = v_imp * tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
        a  = 30.0 * v_imp / T_ramp * tau**2 * (1.0 - tau)**2
    else:
        u  = v_imp * (t - 0.5 * T_ramp)
        v  = v_imp
        a  = 0.0
    return u, v, a

# ── Initial state at t = 0 ───────────────────────────────────────────────────
# Set BC values at t=0, enforce them on u_n, then compute p_0 and a_0 so the
# Velocity-Verlet step has a consistent starting acceleration.
u0, v0, a0 = loading(0.0)
u_bc_top_disp.value = u0
u_bc_top_vel.value  = v0
u_bc_top_acc.value  = a0

fem.set_bc(u_n.x.petsc_vec, bc_u)
u_n.x.scatter_forward()
fem.set_bc(v_n.x.petsc_vec, bc_v)
v_n.x.scatter_forward()

# u_new is the field the UFL forms reference; mirror u_n into it for the IC solve.
u_new.x.array[:] = u_n.x.array
u_new.x.scatter_forward()

A_phi.zeroEntries()
assemble_matrix(A_phi, a_phi, bcs=bc_phi)
A_phi.assemble()
with b_phi.localForm() as loc:
    loc.set(0)
assemble_vector(b_phi, L_phi)
b_phi.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
set_bc(b_phi, bc_phi)
solver_phi.solve(b_phi, p_new.x.petsc_vec)
p_new.x.scatter_forward()
p_old.x.array[:] = p_new.x.array

with F_int_vec.localForm() as loc:
    loc.set(0)
assemble_vector(F_int_vec, F_int_form)
F_int_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
F_int_func.x.scatter_forward()
a_n.x.array[:] = F_int_func.x.array / M_lumped_local
fem.set_bc(a_n.x.petsc_vec, bc_a)
a_n.x.scatter_forward()

# Write initial snapshot at t=0
out_file_name.write_function(p_new, 0.0)
out_file_name_u.write_function(u_new, 0.0)

# ── Main time loop ───────────────────────────────────────────────────────────
for i in range(1, num_steps + 1):
    t_phys = i * dt_val_f

    # A. Update imposed loading (displacement, velocity, acceleration at t_phys)
    u_imp_val, v_imp_val, a_imp_val = loading(t_phys)
    u_bc_top_disp.value = u_imp_val
    u_bc_top_vel.value  = v_imp_val
    u_bc_top_acc.value  = a_imp_val

    if rank == 0 and i % save_freq == 0:
        print(f"step {i}/{num_steps},  t = {t_phys:.4e} s,  u_imp = {u_imp_val:.4e} m", flush=True)

    # B. Verlet displacement predictor: u_{n+1} = u_n + dt*v_n + 0.5*dt^2*a_n
    u_new.x.array[:] = (
        u_n.x.array
        + dt_val_f * v_n.x.array
        + 0.5 * dt_val_f**2 * a_n.x.array
    )
    fem.set_bc(u_new.x.petsc_vec, bc_u)
    u_new.x.scatter_forward()

    # C. Phase-field solve (staggered)
    A_phi.zeroEntries()
    assemble_matrix(A_phi, a_phi, bcs=bc_phi)
    A_phi.assemble()
    with b_phi.localForm() as loc:
        loc.set(0)
    assemble_vector(b_phi, L_phi)
    b_phi.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b_phi, bc_phi)
    solver_phi.solve(b_phi, p_new.x.petsc_vec)
    p_new.x.scatter_forward()

    # D. Internal force F_int(u_{n+1}, p_{n+1})
    with F_int_vec.localForm() as loc:
        loc.set(0)
    assemble_vector(F_int_vec, F_int_form)
    F_int_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    F_int_func.x.scatter_forward()

    # E. Acceleration: a_{n+1} = F_int / M_lumped  (rho baked into M_lumped)
    a_new.x.array[:] = F_int_func.x.array / M_lumped_local
    fem.set_bc(a_new.x.petsc_vec, bc_a)
    a_new.x.scatter_forward()

    # F. Velocity corrector: v_{n+1} = v_n + 0.5*dt*(a_n + a_{n+1})
    v_new.x.array[:] = v_n.x.array + 0.5 * dt_val_f * (a_n.x.array + a_new.x.array)
    fem.set_bc(v_new.x.petsc_vec, bc_v)
    v_new.x.scatter_forward()

    # G. State update (advance n -> n+1) and history field
    u_n.x.array[:] = u_new.x.array
    v_n.x.array[:] = v_new.x.array
    a_n.x.array[:] = a_new.x.array
    p_old.x.array[:] = p_new.x.array
    H_old.interpolate(H_expr)

    # Reaction force (includes inertia; see virtual_work_form)
    v_reac.x.petsc_vec.set(0.0)
    fem.set_bc(v_reac.x.petsc_vec, [bc_bot_rxn])
    v_reac.x.scatter_forward()
    R_bot = domain.comm.gather(fem.assemble_scalar(virtual_work_form), root=0)

    v_reac.x.petsc_vec.set(0.0)
    fem.set_bc(v_reac.x.petsc_vec, [bc_left_rxn])
    v_reac.x.scatter_forward()
    R_left = domain.comm.gather(fem.assemble_scalar(virtual_work_form), root=0)

    if rank == 0:
        B_bot.append([np.sum(R_bot), t_phys])
        B_left.append([np.sum(R_left), t_phys])

    if i % save_freq == 0:
        out_file_name.write_function(p_new, t_phys)
        out_file_name_u.write_function(u_new, t_phys)
        if rank == 0:
            plot_force_disp(B_bot, "bot_rxn", out_file)
            plot_force_disp(B_left, "left_rxn", out_file)

end_time   = time.time()
total_time = end_time - start_time

out_file_name.close()
out_file_name_u.close()

if rank == 0:
    print(f"Simulation completed in {total_time:.2f} s")
    print(f"rank = {rank} done.")