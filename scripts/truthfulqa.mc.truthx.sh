export CUDA_VISIBLE_DEVICES=7

ROOT=/home/xjg/TruthX
EXP_ROOT=$ROOT/myresults
model_path=/home/xjg/checkpoints/mistral-7b-v0.1 # e.g. Llama-2-7b-chat-hf

# two-fold validation
truthx_model1=/home/xjg/myTruthX/mistral/mytruthx_model.fold1_300epoch.pt
truthx_model2=/home/xjg/myTruthX/mistral/mytruthx_model.fold2_300epoch.pt

strength=12.5 # 12.5
layers=10

python3  $ROOT/scripts/truthfulqa_mc_truthx.py \
    --model-path $model_path \
    --truthx-model $truthx_model1 \
    --truthx-model2 $truthx_model2 \
    --two-fold False\
    --data-yaml data/truthfulqa_data_fold1.yaml \
    --edit-strength $strength --top-layers $layers \
    --fewshot-prompting True \
    --output-dir $EXP_ROOT/truthfulqa_mc_truthx/my_mistral-7b-v0.1/fold5