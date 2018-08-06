#!/bin/bash

# Some reasonable bash constraints.
set -xeuo pipefail

# Miniconda Config:
export MINICONDA_VERSION="4.5.4"
export MINICONDA_MD5="164ec263c4070db642ce31bb45d68813"
export MINICONDA_INSTALLER="${HOME}/miniconda.sh"
export MINICONDA_DIR="${HOME}/miniconda"
export MINICONDA_URL="https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-MacOSX-x86_64.sh"

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
