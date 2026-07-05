export ASCEND_RT_VISIBLE_DEVICES=""4,5,6,7""
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=""AIV""
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=""expandable_segments:True""
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD


vllm serve /mnt/share/weight/Qwen3-Next-80B-A3B-Instruct-w8a8-mtp \
       --served-model-name ""Qwen3-80B"" \
       --host 0.0.0.0\
       --trust-remote-code \
       --enable-expert-parallel \
       --max-num-seqs 48 \
       --max-model-len 40960 \
       --max-num-batched-tokens 40960 \
       --tensor-parallel-size 2 \
       --data-parallel-size 2 \
       --disable-custom-all-reduce \
       --port 8800 \
       --block-size 32 \
       --enable_chunked_prefill \
       --enable-prefix-caching \
       --enforce-eager \
       --async-scheduling \
       --compilation-config '{""cudagraph_mode"": ""FULL_DECODE_ONLY""}' \
       --gpu-memory-utilization 0.9 \
       --additional-config '{""pa_shape_list"":[48,64,72,80], ""weight_prefetch_config"":{""enabled"":true}}'

