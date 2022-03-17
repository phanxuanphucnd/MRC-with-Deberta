#Intensive module
export SQUAD_DIR=data/bdi-mrc
export TRAIN_FILE=train-v2.0.json
export DEV_FILE=dev-v2.0.json
python ./run_av.py \
    --model_type deberta-v3 \
    --model_name_or_path models/bdi-debertav3-xsmall \
    --do_eval \
    --do_lower_case \
    --version_2_with_negative \
    --train_file $SQUAD_DIR/$TRAIN_FILE \
    --predict_file $SQUAD_DIR/$DEV_FILE \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --max_query_length=64 \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=64 \
    --warmup_steps=814 \
    --output_dir models/bdi-mrc/debertav3 \
    --eval_all_checkpoints \
    --save_steps 5000 \
    --n_best_size=20 \
    --max_answer_length=50 \
    --overwrite_output_dir \
    --overwrite_cache
