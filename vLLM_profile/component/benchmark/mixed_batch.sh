
BATCH1="8192:4"
BATCH2="512:4"

python benchmark_flash_attn_mixed_batch.py --batch-configs "$BATCH1" "$BATCH2" "${BATCH1} ${BATCH2}" --output-dir mixed_batch_${BATCH1}_${BATCH2}