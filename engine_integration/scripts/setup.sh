#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)

check_uv() {
    if ! command -v uv &> /dev/null; then
        echo "Error: uv is not installed"
        echo "Please install uv first, e.g., 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
        exit 1
    fi
}

install_requirements() {
    pushd "$KVCACHED_DIR"
    uv pip install -r requirements.txt
    popd
}

setup_vllm() {
    pushd "$ENGINE_DIR"

    git clone -b v0.8.4 https://github.com/vllm-project/vllm.git vllm-v0.8.4
    cd vllm-v0.8.4

    uv venv --python=python3.11
    source .venv/bin/activate
    uv pip install --upgrade pip

    # Install requirements for kvcached first to avoid overwriting vLLM's requirements
    install_requirements

    VLLM_USE_PRECOMPILED=1 uv pip install --editable .
    git apply "$SCRIPT_DIR/kvcached-vllm-v0.8.4.patch"

    # Install kvcached after installing VLLM to find the correct torch version
    pushd "$KVCACHED_DIR"
    uv pip install -e . --no-build-isolation
    popd

    deactivate
    popd
}

setup_sglang() {
    pushd "$ENGINE_DIR"

    git clone -b v0.4.6.post2 https://github.com/sgl-project/sglang.git sglang-v0.4.6.post2
    cd sglang-v0.4.6.post2

    uv venv --python=python3.11
    source .venv/bin/activate
    uv pip install --upgrade pip

    # Install requirements for kvcached first to avoid overwriting sglang's requirements
    install_requirements

    uv pip install -e "python[all]"
    git apply "$SCRIPT_DIR/kvcached-sglang-v0.4.6.post2.patch"

    # Install kvcached after install sglang to find the correct torch version
    pushd "$KVCACHED_DIR"
    uv pip install -e . --no-build-isolation
    popd


    deactivate
    popd
}

op=${1:-}

if [ -z "$op" ]; then
    echo "Usage: $0 <vllm|sglang|all>"
    exit 1
fi

# Check for uv before proceeding
check_uv

case "$op" in
    "vllm")
        setup_vllm
        ;;
    "sglang")
        setup_sglang
        ;;
    "all")
        setup_vllm
        setup_sglang
        ;;
    *)
        echo "Error: Unknown option '$op'"
        echo "Usage: $0 <vllm|sglang|all>"
        exit 1
        ;;
esac
