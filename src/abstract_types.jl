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

abstract type Equation end

abstract type BoundaryCondition end
abstract type Grid{dimension} end

abstract type NumericalFlux end

abstract type Reconstruction end

abstract type System end

abstract type TimeStepper end

abstract type SourceTerm end

struct SourceTermBottom <: SourceTerm end
abstract type SourceTermRain <: SourceTerm end
abstract type SourceTermInfiltration <: SourceTerm end

# TODO: Define AbstractSimulator as well?

# TODO: Define all none-abstract types here as well?
