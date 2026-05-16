#!/bin/sh


set -x

echo 'Set cpupower to performance mode'
sudo cpupower frequency-set -g performance > /dev/null
#MOE_QUANT_ONLY=1 SGLANG_CPU_OMP_THREADS_BIND="0-95" python -m sglang.bench_one_batch --batch-size 4 --input 1024 --output 1024 --model /data/farshad/github/sglang/quant_model_dir/Llama-4-Maverick-17B-128E-Instruct-w8g-1/ --trust-remote-code --device cpu --mem-fraction-static 0.8 --tp=1 --quantization w8a8_int8 --max-total-tokens 10000 --result-filename maverick7_10.jsonl --prompt-filename prompt_llama4.json

#--profile             Use Torch Profiler.
#--profile-filename-prefix PROFILE_FILENAME_PREFIX
date="sep2"

logfile_maverick="${date}_maverick_profile.txt"
logfile_scout="${date}_scout_profile.txt"

result_filename_maverick="${date}_maverick.jsonl"
result_filename_scout="${date}_scout.jsonl"

extra="--profile --profile-filename-prefix ${date}"
extra=""

numa_binding="0-42|43-85|86-127|128-170|171-213|214-255"

echo "start" > $logfile_maverick
echo "start" > $logfile_scout

for bs in 1 4 8 12 16 20 26 29 32 43 64 128 148 170; do
    echo "Running batch size: $bs"
    echo "Running batch size: $bs" >> $logfile_maverick
    echo "Running batch size: $bs" >> $logfile_scout

    max_total_tokens=$((bs * 3500))


    #MOE_QUANT_ONLY=1 SGLANG_CPU_OMP_THREADS_BIND=${numa_binding} python -m sglang.bench_one_batch --batch-size $bs --input 1024 --output 1024 --model /data/farshad/github/sglang/quant_model_dir/Llama-4-Scout-17B-16E-Instruct-w8g-1/ --trust-remote-code --device cpu --mem-fraction-static 0.8 --tp=6 --quantization w8a8_int8 --max-total-tokens $max_total_tokens --result-filename $result_filename_scout --prompt-filename prompt_llama4.json ${extra} |tee -a $logfile_scout

    MOE_QUANT_ONLY=1 SGLANG_CPU_OMP_THREADS_BIND=${numa_binding} python -m sglang.bench_one_batch --batch-size $bs --input 1024 --output 1024 --model /data2/farshad/Llama-4-Maverick-17B-128E-Instruct-w8g-1/ --trust-remote-code --device cpu --mem-fraction-static 0.8 --tp=6 --quantization w8a8_int8 --max-total-tokens $max_total_tokens --result-filename $result_filename_maverick --prompt-filename prompt_llama4.json|tee -a $logfile_maverick

done
