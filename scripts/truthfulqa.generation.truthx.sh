export CUDA_VISIBLE_DEVICES=7

ROOT=/home/xjg/TruthX
EXP_ROOT=$ROOT/myresults
model_path=/home/xjg/checkpoints/mistral-7b-v0.1 # e.g. Llama-2-7b-chat-hf

# two-fold validation
truthx_model1=truthx_models/mistral-7b-v0.1/truthx_model.fold1.pt
truthx_model2=truthx_models/mistral-7b-v0.1/truthx_model.fold2.pt

strength=1.0
layers=10

python3  $ROOT/scripts/truthfulqa_generation_truthx.py \
    --model-path $model_path \
    --truthx-model $truthx_model1 \
    --truthx-model2 $truthx_model2 \
    --two-fold True \
    --data-yaml data/truthfulqa_data_fold1.yaml \
    --edit-strength $strength --top-layers $layers  \
    --fewshot-prompting True \
    --output-file $EXP_ROOT/truthfulqa_generation_truthx/test.jsonl
    # --output-file $EXP_ROOT/truthfulqa_generation_truthx/test.jsonl