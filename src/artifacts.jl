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