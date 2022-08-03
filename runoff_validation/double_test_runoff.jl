using Test

using CUDA
using Plots
using ProgressMeter
using NPZ

include("../src/ValidationUtils.jl")


## This script aims to reproduce simulations from
# Fernandez-Pato, Caviedes-Voullieme, Garcia-Navarro (2016) 
# Rainfall/runoff simulation with 2D full shallow water equations: Sensitivity analysis and calibration of infiltration parameters.
# Journal of Hydrology, 536, 496-513. https://doi.org/10.1016/j.jhydrol.2016.03.021



# Run validation experiments
@time run_validation_cases("d_fcg_case_1_1", rain_fcg_1_1)
#@time run_validation_cases("d_fcg_case_1_2", rain_fcg_1_2)
#@time run_validation_cases("d_fcg_case_1_3", rain_fcg_1_3)
#@time run_validation_cases("d_fcg_case_1_4", rain_fcg_1_4)
#@time run_validation_cases("d_fcg_case_1_5", rain_fcg_1_5)

#@time run_validation_cases("d_fcg_case_2_100",  rain_fcg_1_1, topography=2, x0 = 100)
#@time run_validation_cases("d_fcg_case_2_1900", rain_fcg_1_1, topography=2, x0 = 1900)

#@time run_validation_cases("d_fcg_case_3", rain_fcg_3, topography=3, x0 = 200)

