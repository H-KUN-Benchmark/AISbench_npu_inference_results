
export model_path=""/mnt/share/weight/FLUX.1-dev""
export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=2


export RMSNORM_FUSE=1
export ROPE_FUSE=1
export POSEMB_CACHE=1
export ENABLE_LA=1
export ADALN_FUSE=1
export FAST_GELU=1
export USE_NZ=1
export CV_PARALLEL_LEVEL=2

python inference_flux.py \
       --path ${model_path} \
       --save_path ""./res"" \
       --prompt ""Beautiful illustration of The ocean. in a serene landscape, magic realism, narrative realism, beautiful matte painting, heavenly lighting, retrowave, 4 k hd wallpaper"" \
       --device_id 1 \
       --device ""npu"" \
       --width 512 \
       --height 512 \
       --infer_steps 1 \
       --seed 42 \
       --use_cache"
