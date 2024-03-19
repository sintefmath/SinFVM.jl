using SinSWE
using Test

@test SinSWE.minmod(1,2,3) == 1
@test SinSWE.minmod(-1,-2,-3) == -1
@test SinSWE.minmod(1,-2,3) == 0
@test SinSWE.minmod.([1 -2 -3], [3 -5 2], [4 -3 4]) == [1 -2 0]