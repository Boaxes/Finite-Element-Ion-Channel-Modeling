from pathlib import Path
from time import perf_counter

import matplotlib
import numpy
import pyvista
import ufl
from mpi4py import MPI
from dolfinx import default_scalar_type, fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DOC_DIR = RESULTS_DIR

RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# Manufactured-solution Poisson experiment. See "The Poisson Equation and Its Role in Ion Channel Modeling" (pages 9-10).
def exact_solution(x):
    return 1 + x[0] ** 2 + 2 * x[1] ** 2


def solve_poisson(n, write_outputs=False, stem="fundamentals"):
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, n, n, mesh.CellType.quadrilateral
    )
    V = fem.functionspace(domain, ("Lagrange", 1))

    uD = fem.Function(V)
    uD.interpolate(exact_solution)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, default_scalar_type(-6))

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix=f"Poisson{n}",
    )

    t0 = perf_counter()
    uh = problem.solve()
    solve_time = perf_counter() - t0
    uh.name = "uh"

    V2 = fem.functionspace(domain, ("Lagrange", 2))
    uex = fem.Function(V2, name="u_exact")
    uex.interpolate(exact_solution)

    L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
    error_local = fem.assemble_scalar(L2_error)
    error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

    local_max = numpy.max(numpy.abs(uD.x.array - uh.x.array))
    error_max = domain.comm.allreduce(local_max, op=MPI.MAX)

    if write_outputs and domain.comm.size == 1:
        save_outputs(domain, V, uh, stem)

    return {
        "n": n,
        "h": 1.0 / n,
        "error_L2": error_L2,
        "error_max": error_max,
        "solve_time": solve_time,
    }


def save_outputs(domain, V, uh, stem):
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)

    topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    mesh_plotter = pyvista.Plotter(off_screen=True, window_size=(1000, 700))
    mesh_plotter.add_mesh(grid, show_edges=True)
    mesh_plotter.view_xy()
    mesh_plotter.screenshot(RESULTS_DIR / f"{stem}_mesh.png")
    mesh_plotter.close()

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = uh.x.array.real
    u_grid.set_active_scalars("u")

    solution_plotter = pyvista.Plotter(off_screen=True, window_size=(1000, 700))
    solution_plotter.add_mesh(u_grid, show_edges=True)
    solution_plotter.view_xy()
    solution_plotter.screenshot(RESULTS_DIR / f"{stem}_solution.png")
    solution_plotter.close()

    warped = u_grid.warp_by_scalar()
    warped_plotter = pyvista.Plotter(off_screen=True, window_size=(1000, 700))
    warped_plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True)
    warped_plotter.screenshot(RESULTS_DIR / f"{stem}_warped.png")
    warped_plotter.close()

    filename = RESULTS_DIR / stem
    try:
        with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
            vtx.write(0.0)
    except Exception as exc:
        if domain.comm.rank == 0:
            print(f"Skipping VTX output for {stem}: {exc}")

    with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)


def save_experiment_figures(results):
    if MPI.COMM_WORLD.rank != 0:
        return

    ns = [entry["n"] for entry in results]
    hs = [entry["h"] for entry in results]
    errors = [entry["error_L2"] for entry in results]
    runtimes = [entry["solve_time"] for entry in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].loglog(hs, errors, marker="o", linewidth=2, color="#1f77b4")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Mesh Width")
    axes[0].set_ylabel("L2 Error")
    axes[0].set_title("L2 Error vs Mesh Width")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(ns, runtimes, marker="o", linewidth=2, color="#d62728")
    axes[1].set_xlabel("Cells Per Side")
    axes[1].set_ylabel("Solve Time (Seconds)")
    axes[1].set_title("Solve Time vs Cells Per Side")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(DOC_DIR / "simple_experiment_convergence.png", dpi=220)
    plt.close(fig)

    coarse_image = plt.imread(RESULTS_DIR / "experiment_n4_solution.png")
    fine_image = plt.imread(RESULTS_DIR / "experiment_n32_solution.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(coarse_image)
    axes[0].set_title("Coarse mesh: 4 x 4 cells")
    axes[0].axis("off")
    axes[1].imshow(fine_image)
    axes[1].set_title("Refined mesh: 32 x 32 cells")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(DOC_DIR / "simple_experiment_resolution_compare.png", dpi=220)
    plt.close(fig)


def print_experiment_table(results):
    if MPI.COMM_WORLD.rank != 0:
        return

    print("\nSimple mesh-refinement experiment")
    print(
        f"{'N':>6} {'h':>10} {'L2 error':>14} {'max error':>14} {'solve time (s)':>16}"
    )
    for entry in results:
        print(
            f"{entry['n']:>6d} {entry['h']:>10.5f} {entry['error_L2']:>14.2e} "
            f"{entry['error_max']:>14.2e} {entry['solve_time']:>16.4f}"
        )


def main():
    baseline = solve_poisson(8, write_outputs=True, stem="fundamentals")

    if MPI.COMM_WORLD.rank == 0:
        print(f"Baseline Error_L2 : {baseline['error_L2']:.2e}")
        print(f"Baseline Error_max : {baseline['error_max']:.2e}")

    experiment_results = []
    for n in [4, 8, 16, 32]:
        stem = f"experiment_n{n}" if n in {4, 32} else f"experiment_data_n{n}"
        experiment_results.append(
            solve_poisson(n, write_outputs=n in {4, 32}, stem=stem)
        )

    print_experiment_table(experiment_results)
    save_experiment_figures(experiment_results)


if __name__ == "__main__":
    main()
