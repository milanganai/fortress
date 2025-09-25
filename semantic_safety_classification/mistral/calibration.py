from scipy.spatial import distance
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


with open('embeddings_train_test_data.pkl', 'rb') as f:
    embeddings = pickle.load(f)

train_embed = embeddings['train_data']

with open('failures.pkl', 'rb') as f:
    failures = pickle.load(f)

class_descriptions = [fail['label'] for fail in failures]


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

try:
    with open('embeddings_failures.pkl', 'rb') as f:
        class_embeddings = pickle.load(f)
except:
    print('HAVE TO QUERY MODEL FOR FAILURE EMBEDDINGS')
    class_embeddings = create_embeddings_from_text(class_descriptions)
    with open('embeddings_failures.pkl', 'wb') as f:
        pickle.dump(class_embeddings, f)

all_dist = {}
for f in failures:
    all_dist[f['label']] = []

def compute_similarity(query_vector, embeddings):
    #distances = []
    for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        #distances.append({"distance":dist, "index": index})
        all_dist[failures[index]['label']].append(dist)

        #print(failures[index]['label'])
        #print(dist)


for index, review in enumerate(train_embed):
    compute_similarity(train_embed[index], class_embeddings)

for fail in all_dist.keys():
    all_dist[fail].sort()
    print(fail)
    percentiles = [0.0, 0.05, 0.1, 0.25, 0.3, 0.4, 0.5, 1.0]
    for p in percentiles:
        l = int(p*(len(all_dist[fail])-1))
        print(p, all_dist[fail][l])
    print()

with open('safe_distances.pkl', 'wb') as f:
    pickle.dump(all_dist, f)

