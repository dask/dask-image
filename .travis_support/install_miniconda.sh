#!/bin/bash

# Some reasonable bash constraints.
set -xeuo pipefail

# Miniconda Config:
export MINICONDA_VERSION="4.8.2"
export MINICONDA_MD5="e0320c20ea13d04407424ecf57b70eaf"
export MINICONDA_INSTALLER="${HOME}/miniconda.sh"
export MINICONDA_DIR="${HOME}/miniconda"
export MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py37_${MINICONDA_VERSION}-MacOSX-x86_64.sh"

# Install Miniconda.
curl -L "${MINICONDA_URL}" > "${MINICONDA_INSTALLER}"
openssl md5 "${MINICONDA_INSTALLER}" | grep "${MINICONDA_MD5}"
bash "${MINICONDA_INSTALLER}" -b -p "${MINICONDA_DIR}"
rm -f "${MINICONDA_INSTALLER}"

# Activate conda.
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
conda activate

# Configure conda.
conda config --set show_channel_urls true
conda config --set auto_update_conda false
conda config --set add_pip_as_python_dependency true

# Unset all bash constraints.
set +xeuo pipefail
