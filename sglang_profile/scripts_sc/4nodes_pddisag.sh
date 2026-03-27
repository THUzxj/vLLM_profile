
MAX_RUNNING_REQUESTS_DECODE=128 \
BASE_PORT=8000 \
PREFILL_NODES=2 \
DECODE_NODES=2 \
ARCHITECTURE="H" \
MEM_FRACTION_STATIC=0.7 \
DP=16 EP=16 TP=16 \
MODEL_PATH="/path_to_model" \
EP_NUM_REDUNDANT_EXPERTS=32 \
MEM_CHUNKED_PREFILL_SIZE=16384 \
PROFILE_TRIGGER_THRESHOLD=1.0 \
UPDATE_MAX_RUNNING_REQUESTS=1 \
TOTAL_ROUNDS=2 \
SKIP_WARMUP=1 \
PROFILE_POLLING_INTERVAL=0.5 \
SEND_INTERVAL=5 \
IL="1000" BS="256" OL="100" PROFILE_STEPS=50 \
bash scripts_sc/launch_and_bench_server_pddisagg.sh \
scripts_sc/launch_server_deepseek-v3_formal_torchprofile_pddisagg_H.sh \
scripts_sc/bench_one_batch_server_profile_metrics_trigger.sh
