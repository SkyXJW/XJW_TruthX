export CUDA_VISIBLE_DEVICES=0

ROOT=/home/xjg/TruthX
EXP_ROOT=$ROOT/myresults

model_path=/home/xjg/checkpoints/mistral-7b-v0.1 #e.g. Llama-2-7b-chat-hf

python3  $ROOT/scripts/truthfulqa_generation.py \
    --model-path $model_path  \
    --fewshot-prompting True \
    --output-file $EXP_ROOT/truthfulqa_generation/mistral-7b-v0.1.jsonl