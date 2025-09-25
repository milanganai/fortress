from scipy.spatial import distance
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


with open('embeddings_train_test_data.pkl', 'rb') as f:
    embeddings = pickle.load(f)

train_embed = embeddings['train_data']

with open('failures.pkl', 'rb') as f:
    failures = pickle.load(f)

class_descriptions = [fail['label'] for fail in failures]

def create_embeddings_from_text(input):
    return model.encode(input)

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
        dist = 1 - model.similarity(query_vector, embedding)
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

