from scipy.spatial import distance
import pickle
import sys

SAFE_TESTING = (sys.argv[1] == "True")

if SAFE_TESTING:
    print('we are performing safety testing (printing True Negative Percent)')
else:
    print('we are performing unsafety testing (printing True Positive Percent)')

with open('embeddings_train_test_data.pkl', 'rb') as f:
    embeddings = pickle.load(f)

safe_test_embed = embeddings['safe_test_data']
unsafe_test_embed = embeddings['unsafe_test_data']

with open('failures.pkl', 'rb') as f:
    failures = pickle.load(f)

class_descriptions = [fail['label'] for fail in failures]

with open('embeddings_failures.pkl', 'rb') as f:
    class_embeddings = pickle.load(f)

with open('safe_distances.pkl', 'rb') as f:
    all_dist = pickle.load(f)
'''
for fail in all_dist.keys():
    print(fail)
    percentiles = [0.0, 0.05, 0.1, 0.25, 0.3, 0.4, 0.5, 1.0]
    for p in percentiles:
        l = int(p*(len(all_dist[fail])-1))
        print(p, all_dist[fail][l])
    print()
'''

percentiles = [0.0, 0.02, 0.05, 0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
N_DIV = 100
percentiles = [str(i) for i in range(N_DIV)] + percentiles
accurate = {}
for p in percentiles:
    accurate[p] = 0

def compute_similarity(query_vector, embeddings):
    #distances = []
    correct = {}
    for p in percentiles:
        correct[p] = SAFE_TESTING

    for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        
        fail = failures[index]['label']
        cc = 0
        for p in percentiles:
            if cc < N_DIV:
                thres = int(p)*all_dist[fail][0]/N_DIV
            else:
                thres = all_dist[fail][int(p*(len(all_dist[fail])-1))]
            cc+=1
            is_safe = dist > thres
            if SAFE_TESTING:
                correct[p] = correct[p] and is_safe
            else:
                correct[p] = correct[p] or (not is_safe)
    
    for p in percentiles:
        accurate[p] += correct[p]
        #distances.append({"distance":dist, "index": index})

        #print(failures[index]['label'])
        #print(dist)

test_embed = unsafe_test_embed
if SAFE_TESTING:
    test_embed = safe_test_embed

for index, review in enumerate(test_embed):
    compute_similarity(test_embed[index], class_embeddings)


for p in percentiles:
    accurate[p] /= len(test_embed)
    print(p, accurate[p])
