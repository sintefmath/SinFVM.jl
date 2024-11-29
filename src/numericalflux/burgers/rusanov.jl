# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

struct Rusanov{EquationType<:Equation} <: NumericalFlux
    eq::EquationType
end

function (rus::Rusanov)(faceminus, faceplus, direction)
    fluxminus = rus.eq(XDIR, faceminus...)
    fluxplus = rus.eq(XDIR, faceplus...)

    eigenvalue_minus = compute_max_abs_eigenvalue(rus.eq, XDIR, faceminus...)
    eigenvalue_plus = compute_max_abs_eigenvalue(rus.eq, XDIR, faceplus...)

    eigenvalue_max = max(eigenvalue_minus, eigenvalue_plus)

    F = 0.5 .* (fluxminus .+ fluxplus) .- 0.5 * eigenvalue_max .* (faceplus .- faceminus)

    return F, eigenvalue_max
end
