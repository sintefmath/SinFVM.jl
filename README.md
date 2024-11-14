# SinFVM.jl
Finite volume solvers with an emphasis on modelling surface water with the shallow water equations.

## Setting up the first time

Setting up the project directly should just be (run from the root of the repository)

```bash
julia --project
] instantiate
```

It is probably a good idea to run all tests first verify everything installed correctly, so from the root of the repository, run

```bash
julia --project test/runtests.jl
```

## Examples
We currently have two main examples for a full simulation scenario, see

  * `examples/urban.jl`
  * `examples/terrain.jl`

