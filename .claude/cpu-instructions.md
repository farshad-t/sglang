# SGLang CPU Development Instructions

This file provides context and guidelines for CPU-specific development in SGLang.

## CPU Architecture Overview

### CPU Backend Detection
- CPU mode is enabled via `SGLANG_USE_CPU_ENGINE=1` environment variable
- Supports x86_64 (with Intel AMX) and arm64 architectures
- Detection functions in `sglang/srt/utils/common.py`:
  - `is_cpu()` - Check if CPU engine is enabled
  - `is_host_cpu_x86()` - Check for x86_64 architecture
  - `is_host_cpu_arm64()` - Check for arm64 architecture
  - `cpu_has_amx_support()` - Check for Intel AMX (Advanced Matrix Extensions)
  - `xpu_has_xmx_support()` - Check for Intel XPU XMX support

### CPU-Specific Build Configuration
- CPU builds use `python/pyproject_cpu.toml` instead of `pyproject.toml`
- Key CPU dependencies:
  - `torch==2.9.0` - CPU-optimized PyTorch
  - `intel-openmp` - Intel OpenMP for CPU parallelism (x86_64 only)
  - `triton==3.5.0` - Triton compiler for CPU kernels
  - `torchao==0.14.1` - Quantization support for CPU

### Intel AMX Backend
- AMX tiles are Intel's matrix multiplication accelerator
- Check availability: `torch._C._cpu._is_amx_tile_supported()`
- AMX backend check: `is_intel_amx_backend_available` (requires sgl-kernel)
- Attention backend: "intel_amx" in `ATTENTION_BACKEND_CHOICES`

### Key Files for CPU Development
- `python/sglang/srt/server_args.py` - Server configuration with CPU options
- `python/sglang/srt/utils/common.py` - CPU detection and utilities
- `python/pyproject_cpu.toml` - CPU-specific dependencies
- `sgl-kernel/pyproject_cpu.toml` - CPU kernel dependencies

## Logging in SGLang

### Standard Logging Pattern
```python
import logging

logger = logging.getLogger(__name__)

# Log levels available:
logger.debug("Detailed information")
logger.info("General information")
logger.warning("Warning messages")
logger.error("Error messages")
logger.critical("Critical errors")
```

### Log Level Configuration
- Set via `--log-level` argument in ServerArgs
- Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Example: `logging.basicConfig(level=getattr(logging, server_args.log_level.upper()))`

### Common Logging Locations
- `python/sglang/cli/serve.py` - Service startup logging
- `python/sglang/cli/utils.py` - CLI utility logging
- `python/sglang/bench_one_batch.py` - Benchmark logging
- `python/sglang/bench_offline_throughput.py` - Throughput benchmark logging
- `python/sglang/jit_kernel/**/*.py` - Kernel API logging

### Kernel API Logging
Special logging for CUDA/CPU kernel operations:
```python
from sglang.kernel_api_logging import debug_kernel_api
```

## CPU Performance Considerations

### Memory Management
- CPU has different memory hierarchy than GPU (L1/L2/L3 cache, main memory)
- Use `device_context()` helper for CPU device management:
  ```python
  with device_context(torch.device("cpu")):
      # CPU-specific operations
  ```

### Attention Backends for CPU
- `intel_amx` - Intel AMX optimized attention (x86_64 with AMX)
- `torch_native` - PyTorch native attention fallback
- `triton` - Triton-compiled kernels for CPU

### Quantization on CPU
- Supported formats: `bitsandbytes`, `gguf`, `w8a8_int8`, `modelopt`
- Use `torchao` for CPU-specific quantization optimizations

## CPU Benchmarking Guidelines

### Environment Setup
```bash
export SGLANG_USE_CPU_ENGINE=1
# For Intel CPUs with AMX:
export OMP_NUM_THREADS=<num_physical_cores>
export KMP_AFFINITY=granularity=fine,compact,1,0
```

### Key Metrics to Track
1. **Throughput** - Tokens per second
2. **Latency** - Time to first token (TTFT) and end-to-end latency
3. **Memory Usage** - Peak memory consumption
4. **CPU Utilization** - Core usage and efficiency
5. **Cache Performance** - L1/L2/L3 cache hit rates

### Profiling CPU Performance
Use torch profiler with CPU activities:
```python
prof = torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
)
```

## Expert Statistics Collection

Expert statistics help understand model behavior and performance:
- Track per-layer execution times
- Monitor memory allocation patterns
- Collect cache hit/miss statistics
- Profile attention mechanism performance
- Analyze quantization accuracy/speed trade-offs

## CPU vs GPU Comparison

When comparing CPU and GPU implementations:
1. Normalize by hardware cost (price per token)
2. Consider different batch sizes (CPU often better at small batches)
3. Measure energy efficiency (Joules per token)
4. Account for memory constraints
5. Test with representative workloads

## Common CPU Issues and Debugging

### Issue: Slow Performance
- Check if AMX is enabled for x86_64
- Verify OpenMP thread count matches physical cores
- Profile to identify bottlenecks (memory bandwidth vs compute)
- Consider quantization to reduce memory pressure

### Issue: Out of Memory
- Reduce batch size or max_new_tokens
- Enable CPU offloading for KV cache
- Use memory-efficient quantization (gguf, bitsandbytes)

### Issue: Inconsistent Results
- Check for non-deterministic operations (set seed)
- Verify quantization settings match between runs
- Ensure proper synchronization in multi-threaded code

## Testing CPU Code

### Running CPU Tests
```bash
export SGLANG_USE_CPU_ENGINE=1
pytest test/ -v -k "cpu"
```

### CPU-Specific Test Considerations
- Mock GPU-specific operations
- Use smaller models for CPU tests
- Test both x86_64 and arm64 code paths (if applicable)
- Verify AMX codepath separately from non-AMX

## Contributing CPU Code

### Code Style
- Follow existing logging patterns
- Add CPU-specific branches with clear comments
- Use `is_cpu()` checks for CPU-specific paths
- Document CPU-specific parameters in docstrings

### Performance Guidelines
- Profile before optimizing
- Use vectorized operations where possible
- Leverage Intel MKL or OpenBLAS for linear algebra
- Consider CPU cache locality in algorithm design

---

## Quick Reference

### Enable CPU Mode
```bash
export SGLANG_USE_CPU_ENGINE=1
```

### Check CPU Support
```python
from sglang.srt.utils.common import is_cpu, cpu_has_amx_support
print(f"CPU mode: {is_cpu()}")
print(f"AMX support: {cpu_has_amx_support()}")
```

### Launch CPU Server
```bash
python -m sglang.launch_server \
    --model-path <model_path> \
    --device cpu \
    --log-level INFO
```

### CPU Benchmark
```bash
export SGLANG_USE_CPU_ENGINE=1
python -m sglang.bench_offline_throughput \
    --model-path <model_path> \
    --log-level INFO
```
