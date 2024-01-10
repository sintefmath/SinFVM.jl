using Plots
using StaticArrays

module SinSWE
using Logging

direction(integer) = Val{integer}

const XDIRT = Val{1}
const YDIRT = Val{2}
const ZDIRT = Val{3}

const XDIR = XDIRT()
const YDIR = YDIRT()
const ZDIR = ZDIRT()

using StaticArrays
using Parameters

struct PeriodicBC 
end



abstract type Grid{dimension} end
struct CartesianGrid{dimension, BoundaryType} <: Grid{dimension}
    ghostcells::SVector{dimension, Int64}
    totalcells::SVector{dimension, Int64}

    boundary::BoundaryType
    extent::SMatrix{dimension, 2, Float64}
end

function CartesianGrid(nx; gc=1, boundary=PeriodicBC(), extent=[0.0 1.0])
    return CartesianGrid(SVector{1, Int64}([gc]),
        SVector{1, Int64}([nx + 2 * gc]),
        boundary, SMatrix{1, 2, Float64}(extent))
end

function cell_centers(grid::CartesianGrid{1}; interior=true)
    @assert interior

    xinterface = collect(LinRange(grid.extent[1, 1], grid.extent[1, 2], grid.totalcells[1] - 2 * grid.ghostcells[1] + 1))
    xcell = xinterface[1:end-1] .+ (xinterface[2]-xinterface[1])/2.0
    return xcell
end

function update_bc!(::PeriodicBC, grid::CartesianGrid{1}, data)
    for ghostcell in 1:grid.ghostcells[1]
        data[ghostcell] = data[end - ghostcell - grid.ghostcells[1] ]
        data[end - (grid.ghostcells[1]-ghostcell)] = data[grid.ghostcells[1] + ghostcell]
    end
end

function update_bc!(grid::CartesianGrid{1}, data)
    update_bc!(grid.boundary, grid, data)
end

function for_each_inner_cell(f, g::CartesianGrid{1}, include_ghostcells=0)
    for i in (g.ghostcells[1]-include_ghostcells+1):(g.totalcells[1]-2*g.ghostcells[1]+include_ghostcells + 1)
        f(i-1, i, i+1)
    end
end
abstract type Equation end
abstract type NumericalFlux end

struct Rusanov{EquationType <: Equation} <: NumericalFlux 
    eq::EquationType
end

function (rus::Rusanov)(left, right)
    flux_left = rus.eq(XDIR, left...)
    flux_right = rus.eq(XDIR, right...)

    eigenvalue_left = compute_max_eigenvalue(rus.eq, XDIR, left...)
    eigenvalue_right = compute_max_eigenvalue(rus.eq, XDIR, right...)

    eigenvalue_max = max(eigenvalue_left, eigenvalue_right)

    F = 0.5 .* (flux_left .+ flux_right) .- 0.5 * eigenvalue_max .* (right .- left);

    return F
end


abstract type Reconstruction end

struct NoReconstruction <: Reconstruction end

function reconstruct!(::NoReconstruction, output_left, output_right, input_conserved, grid::Grid, equation::Equation, ::XDIRT)
    for_each_inner_cell(grid) do ileft, imiddle, iright
        output_left[imiddle] = input_conserved[imiddle]
        output_right[imiddle] = input_conserved[imiddle]
    end
end

function compute_flux!(F::NumericalFlux, output, left, right, grid, equation::Equation, ::XDIRT)
    for_each_inner_cell(grid) do ileft, imiddle, iright
        output[imiddle] += F(left[iright], right[imiddle]) - F(left[imiddle], right[ileft])
    end
end

create_buffer(grid::CartesianGrid{1}, equation::Equation) = zeros(SVector{number_of_conserved_variables(equation), Float64}, grid.totalcells[1])

abstract type System end

struct ConservedSystem{ReconstructionType, NumericalFluxType, EquationType, GridType, BufferType} <: System
    reconstruction::ReconstructionType
    numericalflux::NumericalFluxType
    equation::EquationType
    grid::GridType

    left_buffer::BufferType
    right_buffer::BufferType

    ConservedSystem(reconstruction, numericalflux, equation, grid) = new{
        typeof(reconstruction),
        typeof(numericalflux),
        typeof(equation),
        typeof(grid),
        typeof(create_buffer(grid, equation))
    }(reconstruction, numericalflux, equation, grid, create_buffer(grid, equation), create_buffer(grid,equation))
end

create_buffer(grid, cs::ConservedSystem) = create_buffer(grid, cs.equation)

function add_time_derivative!(output, cs::ConservedSystem, current_state)
    reconstruct!(cs.reconstruction, cs.left_buffer, cs.right_buffer, current_state, cs.grid, cs.equation, XDIR)
    compute_flux!(cs.numericalflux, output, cs.left_buffer, cs.right_buffer, cs.grid, cs.equation, XDIR)
end


struct BalanceSystem{ConservedSystemType <: System, SourceTerm} <: System
    conserved_system::ConservedSystemType
    source_term::SourceTerm
end


function add_time_derivative!(output, bs::BalanceSystem, current_state)
    # First add conserved system (so F_{i+1}-F_i)
    add_time_derivative!(output, bs.conserved_system, current_state)

    # Then add source term
    for_each_inner_cell(bs.conserved_system.grid) do ileft, imiddle, iright
        output[imiddle] += bs.source_term(current_state[imiddle])
    end
end
create_buffer(grid, bs::BalanceSystem) = create_buffer(grid, bs.conserved_system)

abstract type TimeStepper end

struct ForwardEulerStepper <: TimeStepper
end

number_of_substeps(::ForwardEulerStepper) = 1


function do_substep!(output, ::ForwardEulerStepper, system::System, current_state, dt)
    # Reset to zero
    output .= zero(output)

    add_time_derivative!(output, system, current_state)
    output .*= dt
    output .+= current_state
    #@info "End of substep" output current_state
end

struct Simulator
    system
    timestepper
    grid

    substep_outputs::Vector
    current_timestep::MVector{1, Float64}
end

function Simulator(system, timestepper, grid)
    return Simulator(system, timestepper, grid, [create_buffer(grid, system) for _ in 1:number_of_substeps(timestepper) + 1], MVector{1, Float64}([0]))
end

current_state(simulator::Simulator) = simulator.substep_outputs[1]
function set_current_state!(simulator::Simulator, new_state)
    @assert length(simulator.grid.ghostcells) == 1
    
    gc = simulator.grid.ghostcells[1]
    current_state(simulator)[gc+1:end-gc] = new_state
    update_bc!(simulator.grid, current_state(simulator))
end

current_timestep(simulator::Simulator) = simulator.current_timestep[1]
function compute_timestep(::Simulator) 
    # TODO: FIXME!!!
    return 0.001
end

function perform_step!(simulator::Simulator)
    #@info "Before step" simulator.substep_outputs
    simulator.current_timestep[1] = compute_timestep(simulator)
    for substep in 1:number_of_substeps(simulator.timestepper)
        @assert substep + 1 == 2
        do_substep!(simulator.substep_outputs[substep+1], simulator.timestepper, simulator.system, simulator.substep_outputs[substep], simulator.current_timestep[1])
        #@info "before bc" simulator.substep_outputs[substep + 1]
        update_bc!(simulator.grid, simulator.substep_outputs[substep+1])
        #@info "after bc" simulator.substep_outputs[substep + 1]
    end
    #@info "After step" simulator.substep_outputs[1] simulator.substep_outputs[2]
    simulator.substep_outputs[1], simulator.substep_outputs[end] =  simulator.substep_outputs[end], simulator.substep_outputs[1]
end
struct ShallowWaterEquations{T} <: Equation
    ρ::T
    g::T
end

ShallowWaterEquations() = ShallowWaterEquations(1.0, 9.81)

function (eq::ShallowWaterEquations)(::XDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    return  [
        ρ * hu,
        ρ * hu  * hu / h + 0.5 * ρ * g * h^2 ,
        ρ * hu * hv / h
    ]
end

function (eq::ShallowWaterEquations)(::YDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    return  [
        ρ * hv,
        ρ * hu * hv / h,
        ρ * hv  * hv / h + 0.5 * ρ * g * h^2 ,
    ]
end

struct Burgers <: Equation end

(::Burgers)(::XDIRT, u) = 0.5 * u.^2

compute_max_eigenvalue(::Burgers, ::XDIRT, u) = abs(u)
number_of_conserved_variables(::Burgers) = 1


export XDIR, YDIR, ZDIR, ShallowWaterEquations, Burgers, CartesianGrid
end


function run_simulation()
    nx = 64
    grid = SinSWE.CartesianGrid(nx)
    equation = SinSWE.Burgers()
    reconstruction = SinSWE.NoReconstruction()
    numericalflux = SinSWE.Rusanov(equation)
    conserved_system = SinSWE.ConservedSystem(reconstruction, numericalflux, equation, grid)
    timestepper = SinSWE.ForwardEulerStepper()

    x = SinSWE.cell_centers(grid)
    initial = collect(map(z->SVector{1, Float64}([z]), sin.(2*π*x)))
    simulator = SinSWE.Simulator(conserved_system, timestepper, grid)
    # initial_state = SinSWE.current_state(simulator)

    # for i in 1:nx
    #     # @show initial[i-1]
    #     initial_state[i+grid.ghostcells[1]] = initial[i]
    # end
    

    # SinSWE.current_state(simulator)[2:end-1] .= initial
    SinSWE.set_current_state!(simulator, initial)
    @show SinSWE.current_state(simulator)

    t = 0.0

    T = 1.0
    plot(first.(SinSWE.current_state(simulator)))
    while t <= T
        SinSWE.perform_step!(simulator)
        t += SinSWE.current_timestep(simulator)
        #println("$t")
    end

    #print(SinSWE.current_state(simulator))
    plot!(first.(SinSWE.current_state(simulator)))
end

run_simulation()