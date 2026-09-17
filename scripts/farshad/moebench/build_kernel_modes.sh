#!/usr/bin/env bash
# Build the CPU kernel extension carrying fused_experts_cpu's expert_batching_mode, for a
# measurement box that has no sgl-kernel wheel with the argument in it.
#
# The output is a bare `common_ops` extension, not an installed package: importing it is
# what runs TORCH_LIBRARY, which is why bench_moe_cpu.py takes --kernel-so-dir.
#
# It must be built against the SAME interpreter that will run the sweep -- the extension
# links libtorch and encodes the CPython ABI, so a .so built elsewhere will not load.
#
# Two things differ per box and are handled here rather than being left to fail late:
#   - cmake and ninja are absent on some of these boxes; both ship as pip wheels, so a
#     scratch venv provides them without touching the run interpreter's site-packages.
#   - torch 2.13 and newer need C++20 (their ATen headers #error out at 17); older torch
#     builds compile at 17. The standard is picked from the torch version.
#
# Env:
#   PY=<python>      the interpreter that will RUN the sweep       (required)
#   SRC=<dir>        .../python/sglang/kernels/aot/csrc/cpu        (default: from this dir)
#   B=<dir>          build dir                                     (default /tmp/moe_cm_mode2)
#   J=<n>            ninja parallelism                             (default: nproc/4)
set -euo pipefail

PY=${PY:?set PY to the interpreter that will run the sweep}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$(cd "$HERE/../../../python/sglang/kernels/aot/csrc/cpu" && pwd)}
B=${B:-/tmp/moe_cm_mode2}
J=${J:-$(( $(nproc) / 4 ))}

"$PY" -c "import torch" 2>/dev/null || { echo "no torch in $PY" >&2; exit 2; }
TV=$("$PY" -c "import torch;print(torch.__version__)")
TD=$("$PY" -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'share/cmake/Torch'))")
# 2.13 is the first release whose ATen headers require C++20.
STD=$("$PY" -c "import torch;v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]);print(20 if v>=(2,13) else 17)")
echo "torch $TV  ->  -DCMAKE_CXX_STANDARD=$STD"
echo "Torch_DIR=$TD"
echo "src=$SRC  build=$B  j=$J"

# A box that has built this kernel before usually has cmake and ninja as pip wheels inside
# the run interpreter's own env, where they are on no PATH until it is activated.
PATH="$(cd "$(dirname "$PY")" && pwd):$PATH"
export PATH

if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null; then
  T=/tmp/kbuildtools
  echo "cmake/ninja missing on this box; installing them into $T (a scratch venv, so the"
  echo "run interpreter's site-packages is left alone)"
  [ -x $T/bin/pip ] || "$PY" -m venv $T
  $T/bin/pip -q install cmake ninja
  PATH=$T/bin:$PATH
  export PATH
fi
echo "cmake=$(command -v cmake)  ninja=$(command -v ninja)"

rm -rf "$B"; mkdir -p "$B"; cd "$B"
# The CMakeLists calls find_package(Python ...), so the hint is Python_EXECUTABLE --
# Python3_EXECUTABLE is silently IGNORED ("Manually-specified variables were not used"), and
# cmake then picks whatever interpreter it finds first. That builds a .so for the wrong
# CPython ABI, which only shows up later as ModuleNotFoundError: No module named common_ops.
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD="$STD" \
      -DPython_EXECUTABLE="$PY" -DPython_ROOT_DIR="$(dirname "$(dirname "$PY")")" \
      -DTorch_DIR="$TD" "$SRC" > configure.log 2>&1 || {
  echo "CONFIGURE FAILED -- tail of $B/configure.log:" >&2
  tail -30 configure.log >&2
  # The usual cause: no Python.h. cmake reports it as a missing Development.Module.
  grep -q "Development.Module" configure.log && \
    echo "Missing Python development headers: install python3-devel, or point PY at an" \
         "interpreter that has them." >&2
  exit 3
}
echo "configure OK"

ninja -j "$J" > build.log 2>&1 || {
  echo "BUILD FAILED -- first errors:" >&2
  grep -E "^FAILED|error:" build.log | head -15 >&2
  exit 4
}
SO=$(ls "$B"/common_ops*.so)
echo "built $SO"
# Catch an ABI mismatch here rather than as a ModuleNotFoundError three steps later.
TAG=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
case "$SO" in
  *"$TAG") ;;
  *) echo "ABI MISMATCH: built $(basename "$SO") but $PY loads *$TAG. cmake resolved a "\
          "different interpreter than PY." >&2; exit 5 ;;
esac

echo "=== verifying both modes run ==="
MOEBENCH_KERNEL_SO_DIR="$B" "$PY" - <<PY
import sys, torch
sys.path.insert(0, "$B")
import common_ops  # noqa: F401
op = torch.ops.sgl_kernel.fused_experts_cpu
names = [a.name for a in op.default._schema.arguments]
assert "expert_batching_mode" in names, f"built kernel has no expert_batching_mode: {names}"
E, K, N, T, TOPK = 8, 256, 128, 16, 2
g = torch.Generator().manual_seed(1)
hs = (torch.randn((T, K), generator=g) / 8).to(torch.bfloat16)
w1 = (torch.randn((E, 2 * N, K), generator=g) / 24).to(torch.bfloat16)
w2 = (torch.randn((E, K, N), generator=g) / 24).to(torch.bfloat16)
tid = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T)]).to(torch.int32)
tw = (torch.rand((T, TOPK), generator=g) + 0.5) / TOPK
p1 = torch.ops.sgl_kernel.convert_weight_packed(w1)
p2 = torch.ops.sgl_kernel.convert_weight_packed(w2)
# Every argument positionally: only activation and expert_batching_mode carry schema
# defaults, so the optional scale/zero/bias tensors have to be passed as explicit None.
# No backticks in this heredoc -- it is unquoted so \$B expands, which means backticks
# would run as command substitution.
for m in (0, 2, 3):
    out = op(hs, p1, p2, tw, tid, False, 0,
             None, None, None, None, None, None, None, None, None,
             True, "silu", m)
    print("  mode", m, "ran, abs-sum", round(float(out.float().abs().sum()), 4))
PY

echo
echo "MOEBENCH_KERNEL_SO_DIR=$B"
