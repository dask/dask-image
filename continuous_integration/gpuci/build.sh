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

rapids-logger "Check environment variables"
env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Update conda environment"
. /opt/conda/etc/profile.d/conda.sh
rapids-mamba-retry env update -n dask_image -f "$WORKSPACE/continuous_integration/environment-$PYTHON_VER.yml"

rapids-logger "Activate conda env"
conda activate dask_image

rapids-logger "Install cupy"
python -m pip install cupy-cuda112 -f https://pip.cupy.dev/pre

rapids-logger "Install dask-image"
python -m pip install .

rapids-logger "Check Python versions"
python --version

rapids-logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

rapids-logger "Python py.test for dask-image"
py.test $WORKSPACE -v -m cupy --junitxml="$WORKSPACE/junit-dask-image.xml"
