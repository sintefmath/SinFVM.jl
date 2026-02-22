using StaticArrays
using SinFVM

# ============================================================
# Two-layer SWE 1D: equilibrium RHS decomposition + bottom source dispatch check
#   Storage: V = (h1, q1, w, q2), where w = h2 + B
#
# Prints, for each source-term configuration:
#   RHS_full  (measured by tiny micro-step using SinFVM Simulator)
#   RHS_flux  (computed from CentralUpwind face flux divergence only)
#   RHS_res   = RHS_full - RHS_flux  (what the sources/NCP contribute)
#
# Additionally:
#   - checks which evaluate_directional_source_term!(SourceTermBottom, ...) method is DISPATCHED
#   - (optional) runtime debug print confirming the AllTwoLayerSWE-specialized bottom source is CALLED
# ============================================================

# ----------------------------
# OPTIONAL runtime instrumentation (prints once if specialized bottom source is called)
# Set to true if you want runtime proof.
# ----------------------------
const ENABLE_BOTTOM_RUNTIME_DEBUG = true
const _bottom_debug_hit = Ref(false)

if ENABLE_BOTTOM_RUNTIME_DEBUG
    function SinFVM.evaluate_directional_source_term!(
        st::SinFVM.SourceTermBottom,
        output,
        current_state,
        cs::SinFVM.ConservedSystem{<:Any,<:Any,<:Any,<:SinFVM.AllTwoLayerSWE},
        dir::SinFVM.Direction
    )
        if !_bottom_debug_hit[]
            println("DEBUG: AllTwoLayerSWE SourceTermBottom method CALLED (dir=$(dir))")
            _bottom_debug_hit[] = true
        end
        return invoke(SinFVM.evaluate_directional_source_term!,
                      Tuple{SinFVM.SourceTermBottom, typeof(output), typeof(current_state), typeof(cs), typeof(dir)},
                      st, output, current_state, cs, dir)
    end
end

# ----------------------------
# Bottom builder (faces/intersections array -> BottomTopography1D)
# ----------------------------
xwrap(x) = x - floor(x)

function make_bottom_faces_1d_cosine(; nx=128, gc=2, B0=-2.0, A=0.4, m=1)
    backend = SinFVM.make_cpu_backend()
    grid = SinFVM.CartesianGrid(nx; gc=gc, boundary=SinFVM.PeriodicBC())

    xF = SinFVM.cell_faces(grid; interior=false)  # intersections incl ghosts
    Bint = similar(xF)
    @inbounds for i in eachindex(xF)
        x = xwrap(xF[i])
        Bint[i] = B0 + A*cos(2π*m*x)
    end
    bottom = SinFVM.BottomTopography1D(Bint, backend, grid)
    return backend, grid, bottom
end

# ----------------------------
# Equilibrium IC in storage V=(h1,q1,w,q2)
# ----------------------------
function equilibrium_ic(; h10=1.0, w0=-1.0, u0=0.0, min_h=1e-12)
    return (x, Bcell) -> begin
        h1 = max(h10, min_h)
        q1 = h1*u0
        q2 = 0.0
        @SVector [h1, q1, w0, q2]
    end
end

# ----------------------------
# Compute flux divergence from CentralUpwind directly (uses built-in B_face_* lookups)
# rhs[i] = -(F_{i+1/2} - F_{i-1/2}) / dx
# ----------------------------
function flux_divergence_from_CU!(
    rhs::Vector{SVector{4,Float64}},
    eq, cu, bottom, grid,
    U::Vector{SVector{4,Float64}}
)
    ig = grid.ghostcells[1]
    nx = length(U)
    dir = SinFVM.XDIRT()

    # Face fluxes for interior block: indices 1..nx+1 represent
    # faces at left of cell 1, between cells, right of cell nx.
    Fface = Vector{SVector{4,Float64}}(undef, nx+1)

    @inbounds for f in 1:(nx+1)
        im = (f == 1)    ? nx : f-1
        ip = (f == nx+1) ? 1  : f

        Vm = U[im]
        Vp = U[ip]

        # face is the RIGHT face of cell im
        gidx_left = ig + im
        Bface = SinFVM.B_face_right(bottom, gidx_left, dir)

        F, _ = cu(eq, Vm, Vp, dir, Bface)
        Fface[f] = SVector{4,Float64}(F)
    end

    xFint = SinFVM.cell_faces(grid; interior=true)
    dx = xFint[2] - xFint[1]

    @inbounds for i in 1:nx
        rhs[i] = -(Fface[i+1] - Fface[i]) / dx
    end

    return rhs
end

# ----------------------------
# Utility norms
# ----------------------------
maxabs(v::Vector{SVector{4,Float64}}, k::Int) = maximum(i -> abs(v[i][k]), eachindex(v))

# ----------------------------
# Check which bottom source term method DISPATCHES for this ConservedSystem
# ----------------------------
function check_bottom_source_dispatch(cs)
    bottom_src = SinFVM.SourceTermBottom()
    dir = SinFVM.XDIRT()

    meth = which(SinFVM.evaluate_directional_source_term!,
                 (typeof(bottom_src), Any, Any, typeof(cs), typeof(dir)))

    println("---- bottom source dispatch check ----")
    println("ConservedSystem type: ", typeof(cs))
    println("@which evaluate_directional_source_term!(SourceTermBottom, ..., cs, XDIR) =>")
    println("  ", meth)

    sigstr = string(meth.sig)
    println("Dispatched signature: ", sigstr)

    if occursin("AllTwoLayerSWE", sigstr)
        println("OK: Using AllTwoLayerSWE-specialized SourceTermBottom method.")
    else
        error("WRONG bottom SourceTermBottom method dispatched! Expected AllTwoLayerSWE-specialized method, got: $sigstr")
    end

    return nothing
end

# ============================================================
# MAIN
# ============================================================
function main()
    # --- parameters
    nx = 128
    gc = 2
    B0 = -2.0
    A  = 0.4
    m  = 1

    ρ1 = 0.98
    ρ2 = 1.0
    g  = 10.0

    h10 = 1.0
    w0  = -1.0
    u0  = 0.0

    # tiny probe time (estimate instantaneous RHS)
    tprobe = 1e-8

    # --- build bottom/grid/equation/flux/recon
    backend, grid, bottom = make_bottom_faces_1d_cosine(nx=nx, gc=gc, B0=B0, A=A, m=m)
    eq = SinFVM.TwoLayerShallowWaterEquations1D(bottom; ρ1=ρ1, ρ2=ρ2, g=g)

    reconstruction = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1))
    cu = SinFVM.CentralUpwind(eq)

    # --- IC
    xC = SinFVM.cell_centers(grid)
    Bcell = SinFVM.collect_topography_cells(bottom, grid; interior=true)
    ic = equilibrium_ic(h10=h10, w0=w0, u0=u0)
    U0 = [ic(xC[i], Bcell[i]) for i in eachindex(xC)]

    # --- source terms
    bottom_src = SinFVM.SourceTermBottom()
    ncp_src    = SinFVM.SourceTermNonConservative()

    cases = [
        ("bottom only", [bottom_src]),
        ("ncp only",    [ncp_src]),
        ("both",        [bottom_src, ncp_src]),
    ]

    println("---- RHS decomposition at equilibrium start ----")
    println("nx=$nx gc=$gc | B0=$B0 A=$A m=$m")
    println("h10=$h10 w0=$w0 u0=$u0 | ρ1=$ρ1 ρ2=$ρ2 g=$g")
    println("tprobe=$tprobe")
    println("reconstruction = LinearLimiter(Minmod(1))")
    println("NOTE: RHS_flux is computed from CU divergence only; RHS_residual = RHS_full - RHS_flux\n")

    for (name, sources) in cases
        println("\n====================")
        println("CASE: $name")
        println("====================")

        # Build a valid system (SinFVM requires a matching bottom source term if B is nonzero)
        cs  = SinFVM.ConservedSystem(backend, reconstruction, cu, eq, grid, sources)
        check_bottom_source_dispatch(cs)

        sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=0.6)

        # set IC
        SinFVM.set_current_state!(sim, U0)

        # full RHS by micro-step
        st0 = SinFVM.current_interior_state(sim)
        SinFVM.simulate_to_time(sim, tprobe)
        st1 = SinFVM.current_interior_state(sim)

        rhs_full = Vector{SVector{4,Float64}}(undef, length(U0))
        @inbounds for i in eachindex(U0)
            rhs_full[i] = @SVector [
                (st1.h1[i] - st0.h1[i]) / tprobe,
                (st1.q1[i] - st0.q1[i]) / tprobe,
                (st1.w[i]  - st0.w[i])  / tprobe,
                (st1.q2[i] - st0.q2[i]) / tprobe
            ]
        end

        # flux divergence RHS (from CU only) on the SAME initial state
        rhs_flux = [@SVector [0.0, 0.0, 0.0, 0.0] for _ in 1:length(U0)]
        flux_divergence_from_CU!(rhs_flux, eq, cu, bottom, grid, U0)

        # residual = (sources + NCP machinery) contribution
        rhs_res = Vector{SVector{4,Float64}}(undef, length(U0))
        @inbounds for i in eachindex(U0)
            rhs_res[i] = rhs_full[i] - rhs_flux[i]
        end

        println("maxabs RHS full:     ", (maxabs(rhs_full,1), maxabs(rhs_full,2), maxabs(rhs_full,3), maxabs(rhs_full,4)))
        println("maxabs RHS flux:     ", (maxabs(rhs_flux,1), maxabs(rhs_flux,2), maxabs(rhs_flux,3), maxabs(rhs_flux,4)))
        println("maxabs RHS residual: ", (maxabs(rhs_res,1),  maxabs(rhs_res,2),  maxabs(rhs_res,3),  maxabs(rhs_res,4)))

        println("\n---- sample cells (first 6) ----")
        for i in 1:min(6, length(U0))
            println("i=$i  full=$(rhs_full[i])  flux=$(rhs_flux[i])  res=$(rhs_res[i])")
        end
    end
end

main()
