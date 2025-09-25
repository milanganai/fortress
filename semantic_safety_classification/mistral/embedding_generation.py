import pickle
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda")
model = "intfloat/e5-mistral-7b-instruct"
_tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
_model = AutoModel.from_pretrained(model, torch_dtype=torch.float16).to(device)
_device = _model.device


with open('scene_description.pkl', 'rb') as f:
    template_scene_description = pickle.load(f)


scene_description = {'train_data': [], 'safe_test_data': [], 'unsafe_test_data': []}
with open('train_test_data.pkl', 'rb') as f:
    data_combos = pickle.load(f)

for data_split, combos in data_combos.items():
    scene_description[data_split] = []
    for combo in combos:
        concept_string = ""
        for concept in combo:
            concept_string += concept + "\n"
        scene = template_scene_description.replace("${CONCEPTS}",concept_string)
        scene_description[data_split].append(scene)
    print(scene_description[data_split][-1])


def create_embeddings_from_text(input):
    vectors = []
    for prompt in input:
        encoded_input = _tokenizer(
            [prompt], return_attention_mask=False, padding=False, truncation=True
        )
        encoded_input["input_ids"][0] += [_tokenizer.eos_token_id]
        encoded_input = _tokenizer.pad(
            encoded_input,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(_device)
        output = _model(**encoded_input)
        embedding = output.last_hidden_state[:, -1, :]
        embedding = embedding.squeeze()
        embedding = F.normalize(embedding, dim=0).cpu().detach().numpy()
        vectors.append(embedding)
    return vectors


embeddings = {'train_data': [], 'safe_test_data': [], 'unsafe_test_data': []}
for data_split, description in scene_description.items():
    embeddings[data_split] = create_embeddings_from_text(description)
    print(len(embeddings[data_split]))
    #print(embeddings[data_split][ 0])
    #print(embeddings[data_split][-1])
    

with open('embeddings_train_test_data.pkl', 'wb') as f:
    pickle.dump(embeddings, f)




