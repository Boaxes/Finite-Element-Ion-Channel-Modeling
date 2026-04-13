from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import gmsh
import matplotlib
import numpy as np
import pyvista
import ufl
from mpi4py import MPI

from dolfinx import default_scalar_type, fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as gmshio


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------
# Verification problem: 3D Gaussian approximation to Holst's
# single-charge formula on a cube.
# ------------------------------------------------------------

BOX_HALF_WIDTH = 1.0
KAPPA = 2.0
SIGMA_3D = 0.10
LINE_START_RADIUS = 0.05


def exact_yukawa_values(radius: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    safe_radius = np.maximum(radius, 1.0e-14)
    return np.exp(-kappa * safe_radius) / safe_radius


def exact_yukawa_on_points(x, kappa: float = KAPPA):
    radius = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    values = exact_yukawa_values(radius, kappa=kappa)
    return values.astype(default_scalar_type, copy=False)


def gaussian_density_expr_3d(domain, sigma: float):
    x = ufl.SpatialCoordinate(domain)
    radius_sq = x[0] ** 2 + x[1] ** 2 + x[2] ** 2
    normalization = 1.0 / ((2.0 * np.pi * sigma**2) ** 1.5)
    return normalization * ufl.exp(-radius_sq / (2.0 * sigma**2))


def save_cube_outputs(domain, V, uh, stem: str):
    if domain.comm.size != 1 or domain.comm.rank != 0:
        return

    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    mesh_slice = grid.slice(normal="z", origin=(0.0, 0.0, 0.0))

    mesh_plotter = pyvista.Plotter(off_screen=True, window_size=(1100, 800))
    mesh_plotter.set_background("white")
    mesh_plotter.add_mesh(mesh_slice, show_edges=True, color="#d9d9d9", line_width=1)
    mesh_plotter.view_xy()
    mesh_plotter.camera.zoom(1.25)
    mesh_plotter.screenshot(RESULTS_DIR / f"{stem}_mesh.png")
    mesh_plotter.close()

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["potential"] = uh.x.array.real
    u_grid.set_active_scalars("potential")

    solution_slice = u_grid.slice(normal="z", origin=(0.0, 0.0, 0.0))
    scalar_values = solution_slice["potential"]
    clim = [float(np.min(scalar_values)), float(np.max(scalar_values))]

    solution_plotter = pyvista.Plotter(off_screen=True, window_size=(1100, 800))
    solution_plotter.set_background("white")
    solution_plotter.add_mesh(
        solution_slice,
        show_edges=True,
        cmap="viridis",
        clim=clim,
        scalar_bar_args={"title": "Potential"},
    )
    solution_plotter.view_xy()
    solution_plotter.camera.zoom(1.35)
    solution_plotter.screenshot(RESULTS_DIR / f"{stem}_solution.png")
    solution_plotter.close()

    filename = RESULTS_DIR / stem
    try:
        with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
            vtx.write(0.0)
    except Exception as exc:
        print(f"Skipping VTX output for {stem}: {exc}")

    with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)


def save_gaussian_line_comparison(
    domain,
    V,
    uh,
    *,
    kappa: float = KAPPA,
    start_radius: float = LINE_START_RADIUS,
    end_radius: float = BOX_HALF_WIDTH,
):
    if domain.comm.size != 1 or domain.comm.rank != 0:
        return

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["potential"] = uh.x.array.real
    u_grid.set_active_scalars("potential")

    line = pyvista.Line((start_radius, 0.0, 0.0), (end_radius, 0.0, 0.0), resolution=350)
    sampled = line.sample(u_grid)
    radius = np.linalg.norm(sampled.points, axis=1)
    phi_exact = exact_yukawa_values(radius, kappa=kappa)
    phi_numeric = sampled["potential"]
    abs_error = np.abs(phi_numeric - phi_exact)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(radius, phi_exact, linewidth=2.5, color="#111111", label="Holst formula")
    axes[0].plot(
        radius,
        phi_numeric,
        linestyle="--",
        linewidth=2.0,
        color="#1f77b4",
        label="Numerical solution",
    )
    axes[0].set_xlabel("Radius r")
    axes[0].set_ylabel("Potential")
    axes[0].set_title("Comparison Along the x-Axis")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(radius, abs_error, linewidth=2.0, color="#d62728")
    axes[1].set_xlabel("Radius r")
    axes[1].set_ylabel("Absolute Error")
    axes[1].set_title("Error Near and Away from the Center")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pb_gaussian_line_compare.png", dpi=220)
    plt.close(fig)


def save_gaussian_experiment_figures(results):
    if MPI.COMM_WORLD.rank != 0:
        return

    hs = [entry["h"] for entry in results]
    errors = [entry["relative_L2"] for entry in results]
    runtimes = [entry["solve_time"] for entry in results]
    cells = [entry["num_cells"] for entry in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].loglog(hs, errors, marker="o", linewidth=2, color="#1f77b4")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Mesh Width h")
    axes[0].set_ylabel("Relative L2 Error")
    axes[0].set_title("Full-Domain Error vs Mesh Width")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(cells, runtimes, marker="o", linewidth=2, color="#d62728")
    axes[1].set_xlabel("Number of Tetrahedra")
    axes[1].set_ylabel("Solve Time (Seconds)")
    axes[1].set_title("Solve Time vs Problem Size")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pb_gaussian_convergence.png", dpi=220)
    plt.close(fig)

    coarse_image = plt.imread(RESULTS_DIR / "pb_gaussian_n7_solution.png")
    fine_image = plt.imread(RESULTS_DIR / "pb_gaussian_n23_solution.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(coarse_image)
    axes[0].set_title("Coarse mesh: 7 cells per side")
    axes[0].axis("off")
    axes[1].imshow(fine_image)
    axes[1].set_title("Refined mesh: 23 cells per side")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pb_gaussian_resolution_compare.png", dpi=220)
    plt.close(fig)


def solve_linear_pb_gaussian(
    n: int,
    *,
    sigma: float = SIGMA_3D,
    kappa: float = KAPPA,
    half_width: float = BOX_HALF_WIDTH,
    write_outputs: bool = False,
    stem: str = "pb_gaussian_reference",
):
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [(-half_width, -half_width, -half_width), (half_width, half_width, half_width)],
        [n, n, n],
        cell_type=mesh.CellType.tetrahedron,
    )
    V = fem.functionspace(domain, ("Lagrange", 1))

    uD = fem.Function(V)
    uD.interpolate(lambda x: exact_yukawa_on_points(x, kappa=kappa))

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    radius_sq = x[0] ** 2 + x[1] ** 2 + x[2] ** 2
    radius_safe = ufl.sqrt(radius_sq + 1.0e-16)

    rho_sigma = gaussian_density_expr_3d(domain, sigma=sigma)
    source_scale = fem.Constant(domain, default_scalar_type(4.0 * np.pi))
    kappa_sq = fem.Constant(domain, default_scalar_type(kappa**2))

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) + kappa_sq * u * v) * ufl.dx
    L = source_scale * rho_sigma * v * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix=f"PBgaussian{n}",
    )

    t0 = perf_counter()
    uh = problem.solve()
    solve_time = perf_counter() - t0
    uh.name = "Phi_h"

    exact_expr = ufl.exp(-kappa * radius_safe) / radius_safe
    error_form = fem.form(
        ufl.inner(uh - exact_expr, uh - exact_expr) * ufl.dx(metadata={"quadrature_degree": 8})
    )
    error_local = fem.assemble_scalar(error_form)
    error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

    exact_form = fem.form(
        ufl.inner(exact_expr, exact_expr) * ufl.dx(metadata={"quadrature_degree": 8})
    )
    exact_local = fem.assemble_scalar(exact_form)
    exact_L2 = np.sqrt(domain.comm.allreduce(exact_local, op=MPI.SUM))
    relative_L2 = error_L2 / exact_L2

    total_charge_form = fem.form(rho_sigma * ufl.dx)
    total_charge_local = fem.assemble_scalar(total_charge_form)
    total_charge = domain.comm.allreduce(total_charge_local, op=MPI.SUM)

    num_cells = domain.topology.index_map(tdim).size_global

    if write_outputs:
        save_cube_outputs(domain, V, uh, stem)

    return {
        "n": n,
        "h": 2.0 * half_width / n,
        "num_cells": num_cells,
        "sigma": sigma,
        "kappa": kappa,
        "error_L2": float(error_L2),
        "relative_L2": float(relative_L2),
        "solve_time": float(solve_time),
        "total_charge": float(total_charge),
        "solution": uh,
        "space": V,
        "mesh": domain,
    }


def write_gaussian_summary(results):
    if MPI.COMM_WORLD.rank != 0:
        return

    serializable = [
        {
            "n": entry["n"],
            "h": entry["h"],
            "num_cells": entry["num_cells"],
            "sigma": entry["sigma"],
            "error_L2": entry["error_L2"],
            "relative_L2": entry["relative_L2"],
            "solve_time": entry["solve_time"],
            "total_charge": entry["total_charge"],
        }
        for entry in results
    ]
    with (RESULTS_DIR / "pb_gaussian_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)


# ------------------------------------------------------------
# Two-domain experiment: square containing a circular protein
# region with piecewise dielectric coefficients.
# ------------------------------------------------------------

PROTEIN_MARKER = 1
SOLVENT_MARKER = 2
OUTER_BOUNDARY_MARKER = 3

SQUARE_HALF_WIDTH_2D = 1.0
PROTEIN_RADIUS = 0.25
EPSILON_PROTEIN = 2.0
EPSILON_SOLVENT = 80.0
KAPPA_SOLVENT = 2.0
SIGMA_2D = 0.03


def create_two_domain_mesh(
    target_h: float,
    *,
    square_half_width: float = SQUARE_HALF_WIDTH_2D,
    protein_radius: float = PROTEIN_RADIUS,
):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", target_h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", target_h)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    model = gmsh.model
    model.add(f"two_domain_{str(target_h).replace('.', 'p')}")

    if MPI.COMM_WORLD.rank == 0:
        occ = model.occ
        square = occ.addRectangle(
            -square_half_width,
            -square_half_width,
            0.0,
            2.0 * square_half_width,
            2.0 * square_half_width,
        )
        disk = occ.addDisk(0.0, 0.0, 0.0, protein_radius, protein_radius)
        occ.fragment([(2, square)], [(2, disk)])
        occ.synchronize()

        areas = [(tag, occ.getMass(2, tag)) for _, tag in model.getEntities(2)]
        protein_tag = min(areas, key=lambda entry: entry[1])[0]
        solvent_tag = max(areas, key=lambda entry: entry[1])[0]

        model.addPhysicalGroup(2, [protein_tag], PROTEIN_MARKER)
        model.addPhysicalGroup(2, [solvent_tag], SOLVENT_MARKER)

        outer_lines = []
        for dim, tag in model.getEntities(1):
            center = np.array(occ.getCenterOfMass(dim, tag))
            if np.isclose(abs(center[0]), square_half_width) or np.isclose(
                abs(center[1]), square_half_width
            ):
                outer_lines.append(tag)

        model.addPhysicalGroup(1, outer_lines, OUTER_BOUNDARY_MARKER)
        model.mesh.generate(2)

    mesh_data = gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags


def two_domain_coefficients(domain, cell_tags):
    Q = fem.functionspace(domain, ("DG", 0))
    epsilon_r = fem.Function(Q)
    kappa_sq = fem.Function(Q)

    protein_cells = cell_tags.find(PROTEIN_MARKER)
    solvent_cells = cell_tags.find(SOLVENT_MARKER)

    epsilon_r.x.array[protein_cells] = np.full_like(
        protein_cells, EPSILON_PROTEIN, dtype=default_scalar_type
    )
    epsilon_r.x.array[solvent_cells] = np.full_like(
        solvent_cells, EPSILON_SOLVENT, dtype=default_scalar_type
    )

    kappa_sq.x.array[protein_cells] = np.full_like(
        protein_cells, 0.0, dtype=default_scalar_type
    )
    kappa_sq.x.array[solvent_cells] = np.full_like(
        solvent_cells, KAPPA_SOLVENT**2, dtype=default_scalar_type
    )

    return epsilon_r, kappa_sq


def gaussian_density_expr_2d(domain, sigma: float):
    x = ufl.SpatialCoordinate(domain)
    radius_sq = x[0] ** 2 + x[1] ** 2
    normalization = 1.0 / (2.0 * np.pi * sigma**2)
    return normalization * ufl.exp(-radius_sq / (2.0 * sigma**2))


def save_two_domain_mesh(domain, stem: str):
    if domain.comm.size != 1 or domain.comm.rank != 0:
        return

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    plotter = pyvista.Plotter(off_screen=True, window_size=(900, 900))
    plotter.set_background("white")
    plotter.add_mesh(grid, show_edges=True, color="#d9d9d9", line_width=1)
    plotter.view_xy()
    plotter.camera.zoom(1.3)
    plotter.screenshot(RESULTS_DIR / f"{stem}_mesh.png")
    plotter.close()


def save_two_domain_materials(domain, epsilon_r, stem: str):
    if domain.comm.size != 1 or domain.comm.rank != 0:
        return

    tdim = domain.topology.dim
    num_cells_local = domain.topology.index_map(tdim).size_local
    cells = np.arange(num_cells_local, dtype=np.int32)
    domain.topology.create_connectivity(tdim, tdim)
    topology, cell_types, geometry = plot.vtk_mesh(domain, tdim, cells)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.cell_data["epsilon_r"] = epsilon_r.x.array[:num_cells_local].real
    grid.set_active_scalars("epsilon_r")

    plotter = pyvista.Plotter(off_screen=True, window_size=(900, 900))
    plotter.set_background("white")
    plotter.add_mesh(
        grid,
        show_edges=True,
        cmap="plasma",
        scalar_bar_args={"title": "Relative permittivity"},
    )
    plotter.view_xy()
    plotter.camera.zoom(1.3)
    plotter.screenshot(RESULTS_DIR / f"{stem}_materials.png")
    plotter.close()


def save_two_domain_solution(domain, V, uh, stem: str):
    if domain.comm.size != 1 or domain.comm.rank != 0:
        return

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["potential"] = uh.x.array.real
    u_grid.set_active_scalars("potential")

    plotter = pyvista.Plotter(off_screen=True, window_size=(900, 900))
    plotter.set_background("white")
    plotter.add_mesh(
        u_grid,
        show_edges=True,
        cmap="viridis",
        scalar_bar_args={"title": "Potential"},
    )
    plotter.view_xy()
    plotter.camera.zoom(1.3)
    plotter.screenshot(RESULTS_DIR / f"{stem}_solution.png")
    plotter.close()


def save_two_domain_line_profile(domain, V, uh, stem: str):
    if domain.comm.size != 1 or domain.comm.rank != 0:
        return

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["potential"] = uh.x.array.real
    u_grid.set_active_scalars("potential")

    line = pyvista.Line((0.0, 0.0, 0.0), (SQUARE_HALF_WIDTH_2D, 0.0, 0.0), resolution=400)
    sampled = line.sample(u_grid)
    radius = sampled.points[:, 0]
    potential = sampled["potential"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(radius, potential, linewidth=2.2, color="#1f77b4")
    ax.axvline(
        PROTEIN_RADIUS,
        linestyle="--",
        linewidth=1.5,
        color="#d62728",
        label="Interface",
    )
    ax.set_xlabel("x along the line y = 0")
    ax.set_ylabel("Potential")
    ax.set_title("Potential Along a Line Through the Center")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{stem}_line_profile.png", dpi=220)
    plt.close(fig)


def save_two_domain_resolution_compare():
    if MPI.COMM_WORLD.rank != 0:
        return

    coarse_image = plt.imread(RESULTS_DIR / "pb_two_domain_coarse_solution.png")
    fine_image = plt.imread(RESULTS_DIR / "pb_two_domain_refined_solution.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(coarse_image)
    axes[0].set_title("Coarse mesh")
    axes[0].axis("off")
    axes[1].imshow(fine_image)
    axes[1].set_title("Refined mesh")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pb_two_domain_resolution_compare.png", dpi=220)
    plt.close(fig)


def solve_two_domain_pb(
    target_h: float,
    *,
    sigma: float = SIGMA_2D,
    write_outputs: bool = False,
    stem: str = "pb_two_domain_reference",
):
    domain, cell_tags, facet_tags = create_two_domain_mesh(target_h)
    epsilon_r, kappa_sq = two_domain_coefficients(domain, cell_tags)

    V = fem.functionspace(domain, ("Lagrange", 1))
    fdim = domain.topology.dim - 1
    boundary_dofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(OUTER_BOUNDARY_MARKER))
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    rho_sigma = gaussian_density_expr_2d(domain, sigma=sigma)

    a = (ufl.inner(epsilon_r * ufl.grad(u), ufl.grad(v)) + kappa_sq * u * v) * ufl.dx
    L = fem.Constant(domain, default_scalar_type(4.0 * np.pi)) * rho_sigma * v * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix=f"PBtwodomain{str(target_h).replace('.', 'p')}",
    )

    t0 = perf_counter()
    uh = problem.solve()
    solve_time = perf_counter() - t0
    uh.name = "Phi_h"

    num_cells = domain.topology.index_map(domain.topology.dim).size_global
    local_max = np.max(uh.x.array.real)
    max_potential = domain.comm.allreduce(local_max, op=MPI.MAX)

    total_charge_form = fem.form(rho_sigma * ufl.dx)
    total_charge_local = fem.assemble_scalar(total_charge_form)
    total_charge = domain.comm.allreduce(total_charge_local, op=MPI.SUM)

    if write_outputs:
        save_two_domain_mesh(domain, stem)
        save_two_domain_materials(domain, epsilon_r, stem)
        save_two_domain_solution(domain, V, uh, stem)
        save_two_domain_line_profile(domain, V, uh, stem)

        filename = RESULTS_DIR / stem
        try:
            with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
                vtx.write(0.0)
        except Exception as exc:
            print(f"Skipping VTX output for {stem}: {exc}")

        with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(uh)

    return {
        "h": target_h,
        "num_cells": num_cells,
        "solve_time": float(solve_time),
        "max_potential": float(max_potential),
        "total_charge": float(total_charge),
        "solution": uh,
        "space": V,
        "mesh": domain,
    }


def write_two_domain_summary(results):
    if MPI.COMM_WORLD.rank != 0:
        return

    with (RESULTS_DIR / "pb_two_domain_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


def main():
    gaussian_reference = solve_linear_pb_gaussian(
        15,
        write_outputs=True,
        stem="pb_gaussian_reference",
    )

    if MPI.COMM_WORLD.rank == 0:
        print(f"Reference global rel. L2 : {gaussian_reference['relative_L2']:.2e}")
        print(f"Reference total charge   : {gaussian_reference['total_charge']:.6f}")

    gaussian_results = []
    for n in [7, 11, 15, 19, 23]:
        stem = f"pb_gaussian_n{n}" if n in {7, 23} else f"pb_gaussian_data_n{n}"
        gaussian_results.append(
            solve_linear_pb_gaussian(
                n,
                write_outputs=n in {7, 23},
                stem=stem,
            )
        )

    finest_gaussian = gaussian_results[-1]
    save_gaussian_line_comparison(
        finest_gaussian["mesh"],
        finest_gaussian["space"],
        finest_gaussian["solution"],
    )
    save_gaussian_experiment_figures(gaussian_results)
    write_gaussian_summary(gaussian_results)

    if MPI.COMM_WORLD.rank == 0:
        print("\nGaussian-source linearized PB experiment")
        print(
            f"{'n':>6} {'h':>10} {'cells':>10} {'rel L2':>12} {'solve time (s)':>16}"
        )
        for entry in gaussian_results:
            print(
                f"{entry['n']:>6d} {entry['h']:>10.5f} {entry['num_cells']:>10d} "
                f"{entry['relative_L2']:>12.2e} {entry['solve_time']:>16.4f}"
            )

    two_domain_outputs = []
    coarse = solve_two_domain_pb(0.18, write_outputs=True, stem="pb_two_domain_coarse")
    fine = solve_two_domain_pb(0.06, write_outputs=True, stem="pb_two_domain_refined")
    medium = solve_two_domain_pb(0.10, write_outputs=True, stem="pb_two_domain_reference")

    two_domain_outputs.append(
        {
            "h": 0.18,
            "num_cells": coarse["num_cells"],
            "solve_time": coarse["solve_time"],
            "max_potential": coarse["max_potential"],
            "total_charge": coarse["total_charge"],
        }
    )
    two_domain_outputs.append(
        {
            "h": 0.10,
            "num_cells": medium["num_cells"],
            "solve_time": medium["solve_time"],
            "max_potential": medium["max_potential"],
            "total_charge": medium["total_charge"],
        }
    )
    two_domain_outputs.append(
        {
            "h": 0.06,
            "num_cells": fine["num_cells"],
            "solve_time": fine["solve_time"],
            "max_potential": fine["max_potential"],
            "total_charge": fine["total_charge"],
        }
    )

    save_two_domain_resolution_compare()
    write_two_domain_summary(two_domain_outputs)

    if MPI.COMM_WORLD.rank == 0:
        print("\nTwo-domain linearized PB experiment")
        print(
            f"{'h':>8} {'cells':>10} {'max potential':>16} {'solve time (s)':>16}"
        )
        for entry in two_domain_outputs:
            print(
                f"{entry['h']:>8.2f} {entry['num_cells']:>10d} "
                f"{entry['max_potential']:>16.4f} {entry['solve_time']:>16.4f}"
            )


if __name__ == "__main__":
    main()
