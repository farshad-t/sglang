## Steps to start and test Qwen3-moe experiment

- Launch container
```
bash start_sglang_container.sh
pushd /sgl-workspace/sglang
git checkout python/pyproject.toml sgl-kernel/pyproject.toml
git remote add upstream https://github.com/sgl-project/sglang.git
git fetch upstream && git checkout upstream/main
cp python/pyproject_cpu.toml python/pyproject.toml && cp sgl-kernel/pyproject_cpu.toml sgl-kernel/pyproject.toml
pushd python && uv pip install -e . && popd
pushd sgl-kernel && bash build.sh 3.12 cpu && uv pip install . --no-build-isolation --force-reinstall && popd
```

- Test static batchsize
Note: In the `bench_sglang_offline.sh` script, modify the input/output/concurrencies vars in the `bench_sweep` function.
```
bash bench_sglang_offline.sh
```
