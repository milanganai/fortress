#!/bin/bash


export OPENAI_API_KEY="your key"
export VOYAGEAI_API_KEY="your key"

models=(
    "minilm"
    "mistral"
    "mpnet"
    "multilinguale5"
    "openai"
    "qwen"
    "sfr"
    "voyageai"
)

mkdir data

cd _dataset_files
#query reasoner for fail modes
bash run_c_datagen.sh
cd ..

bash prepare.sh

for f in $(seq 1 5); do # fail mode seeds
    mkdir "data/fm_count_ablate_${f}"
    for model in "${models[@]}"; do
        mkdir "data/fm_count_ablate_${f}/${model}"
    done
    for i in $(seq 1 50); do # 1-i fail modes
        cd _dataset_files
        python extract_failures.py $i $f
        cd ..
        for model in "${models[@]}"; do
            cd "${model}"
            # avoid recomputing embeddings 50 times
            mv embeddings_train_test_data.pkl embeddings_train_test_data.pk
            rm *.pkl
            mv embeddings_train_test_data.pk embeddings_train_test_data.pkl
            cp ../_dataset_files/*.pkl .
            cp ../_dataset_files/fmca_run.sh .
            bash "fmca_run.sh" "../data/fm_count_ablate_${f}/${model}/$i.txt"
            cd ..
        done
    done
done
