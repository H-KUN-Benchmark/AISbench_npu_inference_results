export VLLM_USE_V1=1
export HCCL_OP_EXPANSION_MODE=""AIV""
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_NZ=2
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

python -m vllm.entrypoints.openai.api_server \
    --model /mnt/weight/weight/qwen3-30B-w8a8 \
    --served-model-name qwen3-xjn \
    --trust-remote-code \
    --max-num-seqs 256 \
    --max-model-len 37364 \
    --max-num-batched-tokens 16384 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --port 8010 \
    --distributed_executor_backend ""mp"" \
    --no-enable-prefix-caching \
    --async-scheduling True \
    --quantization ascend \
    --compilation-config '{""cudagraph_mode"": ""FULL_DECODE_ONLY""}' \
    --gpu-memory-utilization 0.95
