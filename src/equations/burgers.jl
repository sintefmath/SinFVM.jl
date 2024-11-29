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


struct Burgers <: Equation end

(::Burgers)(::XDIRT, u) = @SVector [0.5 * u .^ 2]

compute_max_abs_eigenvalue(::Burgers, ::XDIRT, u) = abs(first(u))
conserved_variable_names(::Type{Burgers}) = (:u,)
