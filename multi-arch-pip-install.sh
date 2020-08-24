#!/bin/bash

OS_ARCH="$(uname -m)"

REQUIREMENTS_FILE=requirements.txt
# use different, more simplified requirements it inside arm64 docker
if [ "$OS_ARCH" == "aarch64" ]; then
    echo "Using simplified requirements for arm64"
    REQUIREMENTS_FILE=requirements-arm64.txt
else
    echo "Using normal requirements.txt"
fi

pip install -r ${REQUIREMENTS_FILE} -i https://pypi.org/simple --extra-index-url https://${SIT_PYPI_USER}:${SIT_PYPI_PASS}@sit-pypi.herokuapp.com/simple && \
    rm -rf /tmp/pip* /root/.cache
