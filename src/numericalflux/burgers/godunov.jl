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

struct Godunov{EquationType<:Equation} <: NumericalFlux
    eq::EquationType
end

function (god::Godunov)(faceminus, faceplus, direction)
    f(u) = god.eq(XDIR, u...)
    fluxminus = f(max.(faceminus, zero(faceminus)))
    fluxplus = f(min.(faceplus, zero(faceplus)))

    F = max.(fluxminus, fluxplus)
    return F, max(compute_max_abs_eigenvalue(god.eq, XDIR, faceminus...),
        compute_max_abs_eigenvalue(god.eq, XDIR, faceplus...))
end
