import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

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
    return model.encode(input, normalize_embeddings=True)


embeddings = {'train_data': [], 'safe_test_data': [], 'unsafe_test_data': []}
for data_split, description in scene_description.items():
    embeddings[data_split] = create_embeddings_from_text(description)
    print(len(embeddings[data_split]))
    #print(embeddings[data_split][ 0])
    #print(embeddings[data_split][-1])
    

with open('embeddings_train_test_data.pkl', 'wb') as f:
    pickle.dump(embeddings, f)




