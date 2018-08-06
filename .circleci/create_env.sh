#!/bin/bash -l

# Some reasonable bash constraints.
set -xeuo pipefail

# Check CONDA_ENV_TYPE was set.
if [ -z "${CONDA_ENV_TYPE}" ]; then
    echo "Set $CONDA_ENV_TYPE externally."
    exit 1
fi

# Activate conda.
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
conda activate

# Create a temporary directory for the environment.
export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export CONDA_ENV_PREFIX="$(python -c 'import tempfile; print(tempfile.mkdtemp())')"
export CONDA_ENV_PATH="${CONDA_ENV_PREFIX}/${CONDA_ENV_TYPE}"
export CONDA_ENV_SPEC="${SCRIPT_DIR}/environments/${CONDA_ENV_TYPE}.yml"

# Fill the temporary directory.
conda env create -p "${CONDA_ENV_PATH}" -f "${CONDA_ENV_SPEC}"
conda activate "${CONDA_ENV_PATH}"

# Unset all bash constraints.
set +xeuo pipefail
