# Semantic Safety for Real-Time Out-of-Distribution Failure Prevention via Multi-Modal Reasoning

This repository contains the semantic safety implementation for the paper "Real-Time Out-of-Distribution Failure Prevention via Multi-Modal Reasoning" published in CoRL 2025, which presents a framework for detecting potential failure modes in autonomous systems using multi-modal reasoning and embedding-based similarity detection.


## Architecture

### Embedding Models Evaluated

The system evaluates the following embedding models:
- **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`)
- **Mistral** (`mistralai/Mistral-7B-Instruct-v0.1`)
- **MPNet** (`sentence-transformers/all-mpnet-base-v2`)
- **MultilingualE5** (`intfloat/multilingual-e5-large`)
- **OpenAI** (`text-embedding-3-large`)
- **Qwen** (`Qwen/Qwen2-7B-Instruct`)
- **SFR** (`Salesforce/SFR-Embedding-Mistral`)
- **VoyageAI** (`voyage-large-2`)

### System Workflow

1. **Dataset Generation** (`_dataset_files/`):
   - Generates failure modes using OpenAI's reasoning capabilities
   - Creates safe and unsafe scene combinations
   - Structures training and test datasets

2. **Embedding Generation** (per model directory):
   - Converts textual scene descriptions to embeddings
   - Processes both training and test data

3. **Calibration** (per model directory):
   - Computes similarity distributions between safe training data and failure modes
   - Establishes percentile-based safety thresholds

4. **Evaluation** (per model directory):
   - Tests model performance on safe and unsafe scenarios
   - Outputs True Positive and True Negative rates across different thresholds

## Repository Structure

```
foundation_model_anticipate_classification/
├── _dataset_files/                 # Dataset generation and processing
│   ├── context_drone_dataset_generation.py  # Main dataset generation script
│   ├── extract_failures.py        # Failure mode extraction utility
│   ├── fmca_run.sh                # Model evaluation pipeline
│   ├── js_embedding_run.sh        # Embedding generation script
│   └── run_c_datagen.sh           # Dataset generation orchestrator
├── {model_name}/                  # One directory per embedding model
│   ├── calibration.py             # Threshold calibration
│   ├── embedding_generation.py    # Scene embedding generation
│   └── evaluation.py              # Performance evaluation
├── prepare.sh                     # Setup and preparation script
├── run_me.sh                     # Main execution script
└── README.md                     # This file
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- Required Python packages (install via pip):
  ```bash
  pip install openai sentence-transformers scipy numpy pickle-mixin requests torch transformers
  ```


### Quick Start


1. **Set up API keys** (edit `run_me.sh`):
   ```bash
   export OPENAI_API_KEY="your_actual_key"
   export VOYAGEAI_API_KEY="your_actual_key"
   ```

2. **Run the complete pipeline**:
   ```bash
   chmod +x run_me.sh
   ./run_me.sh
   ```

## Detailed Usage

### Manual Execution

If you prefer to run components individually:

1. **Generate Dataset**:
   ```bash
   cd _dataset_files
   bash run_c_datagen.sh
   cd ..
   ```

2. **Prepare Embeddings**:
   ```bash
   bash prepare.sh
   ```

3. **Run Model Evaluation** (for a specific model, e.g., OpenAI):
   ```bash
   cd openai
   python embedding_generation.py
   python calibration.py
   python evaluation.py True   # Safe scenario testing
   python evaluation.py False  # Unsafe scenario testing
   cd ..
   ```

### Dataset Structure

The system uses three types of data:

- **Safe Training Concepts**: Benign objects like "cardboard box", "empty park", "bridge"
- **Safe Test Concepts**: New safe objects for testing generalization
- **Dangerous Test Concepts**: Hazardous scenarios like "pedestrian", "power lines", "tornado"

Scene descriptions combine these concepts with a template describing the drone's monitoring role.

### Failure Mode Categories

The system generates ~50 contextual failure modes including:
- Environmental hazards (fog, storms, high winds)
- Human presence (pedestrians, children, crowds)
- Infrastructure risks (power lines, restricted airspace)
- Dynamic threats (moving vehicles, birds, weather balloons)

## Evaluation Metrics

The system outputs performance across multiple threshold percentiles:
- **True Negative Rate**: Correctly identifying safe scenarios
- **True Positive Rate**: Correctly identifying unsafe scenarios
- **Threshold Sensitivity**: Performance across different safety thresholds (0-100th percentiles)

## Results Structure

Results are saved in `data/fm_count_ablate_{seed}/{model}/` with performance metrics for different numbers of failure modes (1-50) across 5 random seeds.

