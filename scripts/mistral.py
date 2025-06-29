import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from tqdm import tqdm

import pandas as pd
import yaml

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def read_csv(file_name):
    try:
        df = pd.read_csv(file_name)

        # 返回读取的DataFrame
        return df
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
                    "/home/xjg/checkpoints/mistral-7b-v0.1", trust_remote_code=True
                )
    model = AutoModelForCausalLM.from_pretrained(
                    "/home/xjg/checkpoints/mistral-7b-v0.1", trust_remote_code=True
                )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.half()
    model.to(device)

    df = read_csv("/home/xjg/TruthX/data/TruthfulQA.csv")
    data = list(df.T.to_dict().values())
    fold2_data = load_yaml("/home/xjg/TruthX/data/truthfulqa_data_fold2.yaml")["train_set"]

    # 这里的64代表该模型由64个ATT与FFN模块组成
    common_representation_pos = [[] for _ in range(64)]
    common_representation_neg = [[] for _ in range(64)]
    i = 0
    for line in tqdm(data):
        qs = line['Question']
        correct_answer = line['Best Answer']
        incorrect_answer = line['Incorrect Answers']
        
        if i not in fold2_data:
            i += 1
            continue
        i += 1
        
        input_ids1 = tokenizer(line['Question'] + " " + correct_answer, return_tensors="pt").to(device)
        input_ids2 = tokenizer(line['Question'] + " " + incorrect_answer, return_tensors="pt").to(device)

        with torch.no_grad():
            correct_output = model(input_ids1['input_ids'],output_hidden_states=True)
            correct_att_hidden_states = correct_output.att_hidden_states # 多个[batch_size, num_tokens, representation_dim]
            correct_ffn_hidden_states = correct_output.ffn_hidden_states # 多个[batch_size, num_tokens, representation_dim]

            # incorrect_output = model(input_ids2['input_ids'],output_hidden_states=True)
            # incorrect_att_hidden_states = incorrect_output.att_hidden_states
            # incorrect_ffn_hidden_states = incorrect_output.ffn_hidden_states
        
        # 转换 input_ids 为 token
        correct_tokens = tokenizer.convert_ids_to_tokens(input_ids1['input_ids'].squeeze().tolist())
        incorrect_tokens = tokenizer.convert_ids_to_tokens(input_ids2['input_ids'].squeeze().tolist())

        # 将问题转换为token
        question_tokens = tokenizer.convert_ids_to_tokens(tokenizer(line['Question'], return_tensors="pt")['input_ids'].squeeze().tolist())
        question_tokens_len = len(question_tokens)

        # 将回答转换为token
        correct_answer_tokens = tokenizer.tokenize(line['Best Answer'])
        incorrect_answer_tokens = tokenizer.tokenize(line['Incorrect Answers'])

        # 找出同时出现在正确和错误回答中的 token
        common_tokens = set(correct_answer_tokens).intersection(set(incorrect_answer_tokens))

        for token in common_tokens:
            # 在正确和错误回答的 token 列表中找到相同 token 的索引
            correct_index = next((i for i, t in enumerate(correct_answer_tokens) if t == token), None)
            incorrect_index = next((i for i, t in enumerate(incorrect_answer_tokens) if t == token), None)

            # 提取正确回答的internal representation
            for layer in range(len(correct_att_hidden_states)):
                common_representation_pos[2*layer].append(correct_att_hidden_states[layer].squeeze(0)[correct_index+question_tokens_len])
                common_representation_pos[2*layer+1].append(correct_ffn_hidden_states[layer].squeeze(0)[correct_index+question_tokens_len])
            # # 提取错误回答的internal representation
            # for layer in range(len(incorrect_att_hidden_states)):
            #     common_representation_neg[2*layer].append(incorrect_att_hidden_states[layer].squeeze(0)[incorrect_index+question_tokens_len])
            #     common_representation_neg[2*layer+1].append(incorrect_ffn_hidden_states[layer].squeeze(0)[incorrect_index+question_tokens_len])
    
    # 将张量转换为 torch.float32
    common_representation_pos = [torch.stack(item) for item in common_representation_pos]
    common_representation_pos = torch.stack(common_representation_pos)
    common_representation_pos = common_representation_pos.to(torch.float32)
    torch.save(common_representation_pos, "/home/xjg/myTruthX/data/TruthfulQA/data_fold2/mistral/train_common_representations_pos_best.pth")
    
    # # 将张量转换为 torch.float32
    # common_representation_neg = [torch.stack(item) for item in common_representation_neg]
    # common_representation_neg = torch.stack(common_representation_neg)
    # common_representation_neg = common_representation_neg.to(torch.float32)
    # torch.save(common_representation_neg, "/home/xjg/myTruthX/data/TruthfulQA/data_fold2/mistral/train_common_representations_neg_best.pth")

