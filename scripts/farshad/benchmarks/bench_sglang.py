# launch the offline engine
import argparse
import asyncio
import time
import io
import os
import sys


#from PIL import Image
#import requests
import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.utils import get_bool_env_var, set_gpu_proc_affinity
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.managers.io_struct import GenerateReqInput

from sglang.srt.entrypoints.engine import Engine
from sglang_utils import RunnerArgs as BenchArgs
from dataset import Dataset

MODEL_PATH = "/home/sdp/tattafos/LLM-small/meta-llama/Llama-3.1-8B-Instruct"

def run_test(server_args, bench_args, profile=False):
    tp_rank = 0
    #if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
    #    set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)
    #llm = Engine(model_path=server_args.model_path, device="cpu", skip_tokenizer_init=True, mem_fraction_static=server_args.mem_fraction_static)
    llm = Engine(**dataclasses.asdict(server_args))
    print(f"LLM engine initialized with model path: {server_args.model_path}")
    
    print(f"Loading dataset from {bench_args.dataset_path} with {bench_args.num_samples} samples")
    data_obj = Dataset(
        dataset_path=bench_args.dataset_path,
        model_checkpoint_path=server_args.model_path,
        total_sample_count= bench_args.num_samples,
    )
    
    data_obj.loadDataset()
    token_ids_list = [data_obj[index][0] for index in range(bench_args.num_samples)]
    input_lens = [data_obj[index][1] for index in range(bench_args.num_samples)]

    sampling_params = {"temperature": 0.0, "top_p": 1, "top_k": 1, 'max_new_tokens': bench_args.output_len[0]}
    if profile:
        import torch
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        
    for i, token_ids in enumerate(token_ids_list):
        obj = GenerateReqInput(
            input_ids=token_ids,
            sampling_params=sampling_params,
            rid=f"req-{i}",)
        tic = time.time()
        loop = asyncio.get_event_loop()
        if profile:
            prof.step()
        generator = llm.tokenizer_manager.generate_request(obj, None)
        ret = loop.run_until_complete(generator.__anext__())
        #print(f"Generated token ids: {ret['output_ids']}", flush=True)
        #print(f"Request info: {ret['meta_info']}", flush=True)
        #outputs = llm.generate(input_ids=[token_ids], sampling_params=sampling_params)
        e2e = time.time() - tic

        print(f"Req {ret['meta_info']['id']}; Input len {len(token_ids)} completed in {e2e:.2f} seconds; output len {len(ret['output_ids'])}")
    if profile:
        prof.stop()
        prof.export_chrome_trace("profile_trace.json")
        print("Profiling data saved to profile_trace.json")
def main():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)
    
    #print(server_args)
    # Run the test
    run_test(server_args, bench_args, profile=args.profile)

if __name__=="__main__":
    main()