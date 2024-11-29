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

using LazyArtifacts
export datapath_testdata

"""
Returns the path to test data directory.

When called, this function ensures that the test data is downloaded if not already present
in the artifacts directory. The test data is retrieved from a remote source specified in
the project's Artifacts.toml file and stored locally for subsequent use.

# Returns
- `String`: Absolute path to the directory containing test data

# Note
- Automatically downloads required test data if not present locally
- Uses Julia's artifact system for data management
- Download source is specified in the root Artifacts.toml file
"""
function datapath_testdata()
    artifact"sinfvm_testdata"
end
