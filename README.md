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
We currently have a couple ofexamples for a full simulation scenario, see

  * `examples/shallow_water_1d.jl`
  * `examples/urban.jl`
  * `examples/terrain.jl`

## Generating the documentation

Run (note: currently takes a couple of minutes)

    julia --project docs/make.jl

To view the generated documentation, it is probably a good idea to install the `LiveServer` package and view said documentation through that

    julia -e 'using LiveServer; serve(dir="docs/build")'

then point your browser to `localhost:8000`.