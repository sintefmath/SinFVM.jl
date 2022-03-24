# Rumpetroll.jl
Shallow Water solvers in Julia

# Setting up the first time

Start `julia` from the command line

Open terminal and go into pkg repl "]" and `activate .` the relevant project.

Include Vannlinje as a local repo:
1. Add dependency:
`(Rumpetroll) pkg> add "https://git@github.com/sintefmath/Vannlinje.jl"` or 
`(Rumpetroll) pkg> add "git@github.com:sintefmath/Vannlinje.jl"`
2. define where to check out Vannlinje:
`(Rumpetroll) pkg> develop --local Vannlinje`

The git repo Vannlinje is now in your project's `dev/Vannlinje/` and works like any other git repo.


# Run stuff
From the file via Julia REPL in VSCode:
`ctrl+shift+p Julia: Execute file in REPL`
or in Julia:
`julia> include("<path to your file")`
or from command line (standing in Rumpetroll.jl):
`julia --project=. <path to file>`

# If missing packages
If the environment definition has been updated by someone else:
`] activate .` + `] update`

# Compiling CUDA directly:
´´´nvcc -ptx kp07_kernel.cu -o kp07_kernel.ptx´´´

# Conditional CUDA programming:
See here: https://cuda.juliagpu.org/stable/installation/conditional/

# Misc
## Memory layout and indexing conventions
Multidimensional ´Array´s (and thereby also ´CuArray´s) in Julia are column major - same as Fortran, Matlab and R, but unlike C and Python (numpy).
This means that for a 2-dimensional we should use the following indexing:
`A = randn(Nx, Ny)`, which means that the most efficiently iteration would be
´´´
for j=1:Ny
    for i=1:Nx
        A[i, j]
    end
end
´´´