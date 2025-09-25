#!/bin/bash

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

envs=(
    "context_drone"
)

for env in "${envs[@]}"; do
    for model in "${models[@]}"; do
        cd "${model}"
        rm *.pkl
        cp ../_dataset_files/js_embedding_run.sh .
        cp ../_dataset_files/*.pkl .
        bash js_embedding_run.sh
        cd ..
    done
done
