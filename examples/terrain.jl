# # Modelling of a terrain with a bay
# In this example, we will setup a simple model of a terrain with a bay. We will use the Horton infiltration model to simulate the infiltration of water into the soil. The terrain is represented by a grid, and we will use the shallow water equations to simulate the flow of water over the terrain. We will use the `SinFVM` package to setup and solve the model. The model will be solved using the `ForwardEuler` time-stepping method. We will use the `CairoMakie` package to visualize the results of the simulation.

using DelimitedFiles: DelimitedFiles
using NPZ

using SinFVM
using Meshes: Meshes
using Parameters
using Printf
using StaticArrays
using CairoMakie
using CUDA: CUDA
using SparseArrays

include("example_tools.jl")


function naive_infill(terrain)
	terrain_return = copy(terrain)
	for I in eachindex(terrain)
		if terrain[I] < 1e-3
			min_dist = Inf
			min_I = I
			for J in eachindex(terrain)
				if terrain[J] >= 1e-3
					dist = sqrt(sum((Tuple(I) .- Tuple(J)).^2))
					if dist < min_dist
						min_dist = dist
						min_I = J
					end
				end
			end
			terrain_return[I] = -log(min_dist)
		end
	end
	return terrain_return
end
function solve_lapace(terrain, magic_marker = -100.0)
	n, m = size(terrain)
	N = n * m

	# Create sparse matrix for Laplace equation
	I = Int[]
	J = Int[]
	V = Float64[]

	# For each cell
    source = zeros(N)
	for i in 1:n
		for j in 1:m
			
			idx = (j - 1) * n + i

			# Add diagonal term
			push!(I, idx)
			push!(J, idx)

			if terrain[i, j] > 1e-3
				push!(V, 1.0)
                source[idx] = terrain[i, j]
			else
				push!(V, 4.0)

				# Add neighbor connections
				if i > 1
					push!(I, idx)
					push!(J, idx - 1)
					push!(V, -1.0)
                else
                    # Here we have a Neumann boundary condition
                    # U[0] = U[1] (du/dx = 0)
                    # hence we get the equation
                    # -u[0] + 4u[1] - u[2] = 0
                    # now u[0] == u[1] 
                    push!(I, idx)
                    push!(J, idx)
                    push!(V, -1)
				end
				if i < n
					push!(I, idx)
					push!(J, idx + 1)
					push!(V, -1.0)
                else
                    push!(I, idx)
                    push!(J, idx)
                    push!(V, -1)
				end
				if j > 1
					push!(I, idx)
					push!(J, idx - n)
					push!(V, -1.0)
                else
                    push!(I, idx)
                    push!(J, idx)
                    push!(V, -1)
				end
				if j < m
					push!(I, idx)
					push!(J, idx + n)
					push!(V, -1.0)
                else
                    push!(I, idx)
                    push!(J, idx)
                    push!(V, -1)
				end

			end
		end
	end

	sparse_matrix = sparse(I, J, V, N, N)

    return reshape(sparse_matrix \ source, size(terrain))
end

for backend in get_available_backends()
	dataset_base = joinpath(datapath_testdata(), "data", "small")
	terrain = loadgrid(joinpath(dataset_base, "bay.txt"))
	upper_corner = Float64.(size(terrain))
	coarsen_times = 2
	terrain_original = terrain
	terrain = coarsen(terrain, coarsen_times)
	mkpath("figs/bay/")

	terrain_marked = copy(terrain)

	for i in 1:size(terrain_marked, 1)
		for j in 1:size(terrain_marked, 2)
			is_same = true
			for di in -1:1
				for dj in -1:1
					ni = i + di
					nj = j + dj
					if ni >= 1 && ni <= size(terrain_marked, 1) &&
					   nj >= 1 && nj <= size(terrain_marked, 2) &&
					   (di != 0 || dj != 0)
						is_same = is_same && terrain[ni, nj] == terrain[i, j]
					end
				end
			end
			if is_same
				terrain_marked[i, j] = -100.0
			end
		end
	end

	with_theme(theme_latexfonts()) do
		f = Figure()
		ax1 = Axis(f[1, 1])

		#heatmap!(ax1, terrain_marked, label="marked")
		p = heatmap!(ax1, terrain_marked, label = "marked")
		Colorbar(f[1, 2], p)
		display(f)
	end

    # Solve Laplace equation
    terrain_laplace = naive_infill(terrain)

    with_theme(theme_latexfonts()) do
        f = Figure()
        ax1 = Axis(f[1, 1])

        p = heatmap!(ax1, terrain_laplace, label = "Laplace", colorrange = (minimum(terrain_laplace), 2))
        Colorbar(f[1, 2], p)
        display(f)
    end
	with_theme(theme_latexfonts()) do
		f = Figure()
		ax1 = Axis(f[1, 1])
		ax2 = Axis(f[2, 1])

		heatmap!(ax1, terrain_original, label = "original")
		heatmap!(ax2, terrain, label = "Coarsened")
		save("figs/bay/terrain_comparison.png", f, px_per_unit = 2)

	end

	grid_size = size(terrain) .- (5, 5)
	grid = SinFVM.CartesianGrid(grid_size...; gc = 2, boundary = SinFVM.NeumannBC(), extent = [0 upper_corner[1]; 0 upper_corner[2]])
	infiltration = SinFVM.HortonInfiltration(grid, backend)
	#infiltration = SinFVM.ConstantInfiltration(15 / (1000.0) / 3600.0)
	bottom = SinFVM.BottomTopography2D(terrain, backend, grid)
	bottom_source = SinFVM.SourceTermBottom()
	equation = SinFVM.ShallowWaterEquations(bottom; depth_cutoff = 8e-2)
	reconstruction = SinFVM.LinearReconstruction()
	numericalflux = SinFVM.CentralUpwind(equation)
	constant_rain = SinFVM.ConstantRain(15 / (1000.0))
	friction = SinFVM.ImplicitFriction()

	with_theme(theme_latexfonts()) do
		f = Figure(xlabel = "Time", ylabel = "Infiltration")
		ax = Axis(f[1, 1])
		t = LinRange(0, 60 * 60 * 24.0, 10000)
		CUDA.@allowscalar infiltrationf(t) = SinFVM.compute_infiltration(infiltration, t, CartesianIndex(30, 30))
		CUDA.@allowscalar lines!(ax, t ./ 60 ./ 60, infiltrationf.(t))

		save("figs/bay/infiltration.png", f, px_per_unit = 2)
	end

	conserved_system =
		SinFVM.ConservedSystem(backend,
			reconstruction,
			numericalflux,
			equation,
			grid,
			[
				infiltration,
				constant_rain,
				bottom_source,
			],
			friction)
	timestepper = SinFVM.ForwardEulerStepper()
	simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid)

	u0 = x -> @SVector[0.0, 0.0, 0.0]
	x = SinFVM.cell_centers(grid)
	initial = u0.(x)

	SinFVM.set_current_state!(simulator, initial)
	SinFVM.current_state(simulator).h[1:end, 1:end] = bottom_per_cell(bottom)
	T = 1# 24 * 60 * 60.0
	callback_to_simulator = IntervalWriter(step = 10.0, writer = (t, s) -> callback(terrain, SinFVM.name(backend), t, s))

	total_water_writer = TotalWaterVolume(bottom_topography = bottom)
	total_water_writer(0.0, simulator)
	total_water_writer_interval_writer = IntervalWriter(step = 10.0, writer = total_water_writer)

	SinFVM.simulate_to_time(simulator, T; maximum_timestep = 1.0, callback = [callback_to_simulator, total_water_writer_interval_writer])
end
