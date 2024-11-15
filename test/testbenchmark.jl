using BenchmarkTools
using Random

@benchmark x = sin.(rand(1_000_000))