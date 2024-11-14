using SinFVM
using Test

@test SinFVM.minmod(1,2,3) == 1
@test SinFVM.minmod(-1,-2,-3) == -1
@test SinFVM.minmod(1,-2,3) == 0
@test SinFVM.minmod.([1 -2 -3], [3 -5 2], [4 -3 4]) == [1 -2 0]