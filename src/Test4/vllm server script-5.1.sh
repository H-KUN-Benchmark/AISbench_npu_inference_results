export ASCEND_RT_VISIBLE_DEVICES=""""0""""
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=""""AIV""""
export VLLM_ASCEND_ENABLE_FLASHCOMM=0
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=""""expandable_segments:True""""
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD


vllm serve /mnt/share/weight/Qwen3-0.6B-W8A8 \
       --served-model-name """"Qwen3"""" \
       --host 0.0.0.0\
       --trust-remote-code \
       --max-num-seqs 8 \
       --max-model-len 40960 \
       --max-num-batched-tokens 512 \
       --tensor-parallel-size 1 \
       --disable-custom-all-reduce \
       --quantization ascend \
       --port 8900 \
       --block-size 4 \
       --enable_chunked_prefill \
       --enable-prefix-caching \
       --async-scheduling \
       --compilation-config '{""""cudagraph_mode"""": """"FULL_DECODE_ONLY""""}' \
       --gpu-memory-utilization 0.9 \
       --additional-config '{""""pa_shape_list"""":[48,64,72,80], """"weight_prefetch_config"""":{""""enabled"""":true}}'

