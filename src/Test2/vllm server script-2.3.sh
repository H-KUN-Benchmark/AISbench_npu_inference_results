# set the NPU device number
export ASCEND_RT_VISIBLE_DEVICES=0

# Set the operator dispatch pipeline level to 1 and disable manual memory control in ACLGraph
export TASK_QUEUE_ENABLE=1

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
# if os is Ubuntu
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
# if os is openEuler
# export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD


# Enable the AIVector core to directly schedule ROCE communication
export HCCL_OP_EXPANSION_MODE=""AIV""

# Enable FlashComm_v1 optimization when tensor parallel is enabled.
# export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /mnt/share/weight/Qwen3-14B-w8a8 \
  --port 8008 \
  --served-model-name qwen3 \
  --trust-remote-code \
  --async-scheduling \
  --quantization ascend \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-num-seqs 32 \
  --max-model-len 40960 \
  --max-num-batched-tokens 40960 \
  --enable-prefix-caching \
  --compilation-config '{""cudagraph_mode"": ""FULL_DECODE_ONLY"", ""cudagraph_capture_sizes"":[1,2,4,8,16,24,32]}' \
  --additional-config '{""pa_shape_list"":[1,2,4,8,16,32], ""weight_prefetch_config"":{""enabled"":true}}' \
  --block-size 128 \
  --gpu-memory-utilization 0.9

