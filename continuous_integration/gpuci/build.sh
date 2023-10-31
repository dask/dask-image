##############################################
# Dask GPU build and test script for CI      #
##############################################
set -e
NUMARGS=$#
ARGS=$*

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME="$WORKSPACE"

# Switch to project root; also root of repo checkout
cd "$WORKSPACE"

# Determine CUDA release version
export CUDA_REL=${CUDA_VERSION%.*}

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment variables"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Update conda environment"
. /opt/conda/etc/profile.d/conda.sh
gpuci_mamba_retry env update -n dask_image -f "$WORKSPACE/continuous_integration/environment-$PYTHON_VER.yml"

gpuci_logger "Activate conda env"
conda activate dask_image

gpuci_logger "Install cupy"
python -m pip install cupy-cuda112 -f https://pip.cupy.dev/pre

gpuci_logger "Install dask-image"
python setup.py install

gpuci_logger "Check Python versions"
python --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

gpuci_logger "Python py.test for dask-image"
py.test $WORKSPACE -v -m cupy --junitxml="$WORKSPACE/junit-dask-image.xml"
