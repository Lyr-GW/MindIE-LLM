# AGENTS.md

## Cursor Cloud specific instructions

### Overview

MindIE-LLM is a Huawei Ascend NPU LLM inference engine. It is a hybrid C++17/Python 3.12 project.
The C++ daemon (`mindieservice_daemon`) handles scheduling, KV cache, gRPC/HTTP serving.
The Python layer (`mindie_llm/`) handles model loading, text generation, and connector IPC.

**Important:** Full end-to-end inference requires Ascend NPU hardware + CANN toolkit, which are NOT available in Cloud Agent VMs. However, the C++ source can be fully compiled on CPU, and Python CPU unit tests can be run.

### Compiler setup

The VM default `c++`/`cc` alternatives may point to Clang, which fails to link with libstdc++. Switch to GCC:

```bash
sudo update-alternatives --set c++ /usr/bin/g++
sudo update-alternatives --set cc /usr/bin/gcc
```

### Environment variables for running Python code

```bash
export MINDIE_LLM_HOME_PATH="/workspace/output"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=/workspace:/workspace/src/server/tokenizer:/workspace/build/mindie_llm/connector/cpp:/workspace/build/mindie_llm/text_generator/cpp/memory_bridge:/workspace/build/mindie_llm/text_generator/cpp/prefix_tree:/workspace/build/mindie_llm/text_generator/cpp/sampler/cpu_logits_handler
```

### Build commands

- **Third-party libs:** `bash build.sh 3rd --use_cxx11_abi=1`
- **Main source:** `bash build.sh master --use_cxx11_abi=1`
- **DLT (unit tests):** `bash build.sh dlt --use_cxx11_abi=1` (note: mockcpp has ABI linking issues on this VM; main source compiles fine)
- **Clean:** `bash build.sh clean`

### Lint

```bash
ruff check mindie_llm/ --output-format=concise --line-length=120
```

Only 1 pre-existing error in a generated protobuf file (`model_execute_data_pb2.py`).

### Python tests (CPU)

```bash
pytest tests/pythontest/cpu --ignore=tests/pythontest/cpu/runtime -q
```

Note: some test modules (under `runtime/`) import `torch_npu` / Ascend-specific packages and will fail without NPU hardware. The `--ignore=tests/pythontest/cpu/runtime` flag skips those.

### Log directory

The project writes logs to `/var/log/mindie_log/`. Ensure it exists and is writable:

```bash
sudo mkdir -p /var/log/mindie_log && sudo chmod 777 /var/log/mindie_log
```

### Key gotchas

1. The `build.sh` script sources `scripts/build_env.sh` which auto-detects the PyTorch install path from `python3 -c 'import torch ...'`. Make sure torch is installed before building.
2. `MINDIE_LLM_HOME_PATH` must be set to an absolute path for Python tests to pass (it constructs `benchmark_filepath` from it).
3. C++ DLT tests use `mockcpp` which may have ABI compatibility issues with `USE_CXX11_ABI=1`. The main source build works fine.
4. The `mindieservice_daemon` binary exits with code 255 when no valid config is provided — this is normal behavior.
