# Architecture: RL Training Pipeline for Reasoning Tasks

This document presents the architecture of our reinforcement learning training pipeline for language models on procedurally-generated reasoning tasks.

## System Architecture Overview

```mermaid
graph TB
    subgraph "Configuration System"
        CONFIG[YAML Config Files]
        CONFIG -->|merge inheritance| MERGED[Merged Configuration]
    end

    subgraph "Data Generation & Preparation"
        MERGED -->|task_name, size| RG[Reasoning Gym Library]
        RG -->|procedural generation| TASKS[Reasoning Tasks]
        TASKS -->|format questions| FMT[Add XML Answer Tags]
        FMT --> DATASET[HuggingFace Dataset]
    end

    subgraph "Reward System"
        MERGED -->|reward config| REG[Reward Registry]
        REG --> CR[Correctness Reward]
        CR -->|scoring function| VERIFIER[Task Verifiers]
        CR --> RUBRIC[Reward Rubric]
    end

    subgraph "Model Loading"
        MERGED -->|model config| ML[Model Loader]
        ML -->|HuggingFace| BASE[Base LLM]
        ML -->|LoRA config| LORA[LoRA Adapters]
        BASE --> MODEL[Fine-tunable Model]
        LORA -.->|optional| MODEL
    end

    subgraph "Training Environment"
        DATASET --> ENV[Single-Turn Environment]
        RUBRIC --> ENV
    end

    subgraph "RL Training Loop (GRPO)"
        ENV --> GRPO[GRPO Trainer]
        MODEL --> GRPO

        GRPO --> GEN[Generate N Completions]
        GEN --> EXTRACT[Extract Answers<br/>from XML tags]
        EXTRACT --> SCORE[Score with<br/>Task Verifiers]
        SCORE --> REWARD[Compute<br/>Weighted Rewards]
        REWARD --> UPDATE[Policy Update<br/>GRPO Algorithm]
        UPDATE -->|gradient update| MODEL
        UPDATE --> CKPT[Save Checkpoint]
    end

    subgraph "Evaluation"
        CKPT -.->|trained model| BENCH[Benchmark System]
        TASKS -.->|test set| BENCH
        BASE -.->|baseline| BENCH

        BENCH --> PEVAL[Parallel Evaluation<br/>Multiple GPUs]
        PEVAL --> METRICS[Accuracy Metrics]
        METRICS --> VIZ[Visualization<br/>& Analysis]
    end

    style CONFIG fill:#e1f5ff
    style MERGED fill:#b3e0ff
    style RG fill:#ffe1e1
    style DATASET fill:#ffcccc
    style MODEL fill:#d4edda
    style GRPO fill:#fff3cd
    style BENCH fill:#e7d4f7
    style ENV fill:#ffd9b3
```

## Component Detail Diagram

```mermaid
graph LR
    subgraph "Data Pipeline"
        A[Reasoning Gym<br/>Task Generator] -->|questions + answers| B[Dataset Utils]
        B -->|format_question| C[Add XML Instructions]
        C -->|extract_answer| D[Answer Parser]
        D -->|score_answer| E[Task Verifier]
    end

    subgraph "Training Components"
        F[Model Loader] --> G[Base LLM]
        F --> H[LoRA Config]
        H -.->|PEFT| G

        I[Config System] -->|hyperparams| J[GRPO Config]
        I -->|rewards| K[Reward Registry]
        K --> L[Correctness Reward]

        G --> M[GRPO Trainer]
        J --> M
        L --> M
        E --> L
    end

    subgraph "Evaluation Pipeline"
        N[Test Dataset] --> O[Model Inference]
        G -.->|trained/baseline| O
        O --> D
        E --> P[Accuracy Computation]
        P --> Q[Results & Charts]
    end

    style A fill:#ffe1e1
    style G fill:#d4edda
    style M fill:#fff3cd
    style O fill:#e7d4f7
```

## Training Loop Detail

```mermaid
sequenceDiagram
    participant Env as Training Environment
    participant Model as Language Model
    participant Verifier as Task Verifier
    participant GRPO as GRPO Algorithm

    loop For each training batch
        Env->>Model: Sample questions from dataset
        Model->>Model: Generate N completions per question
        Model->>GRPO: Return completions

        loop For each completion
            GRPO->>GRPO: Extract answer from XML tags
            GRPO->>Verifier: Verify answer correctness
            Verifier->>GRPO: Return score (0.0 to 1.0)
            GRPO->>GRPO: Compute weighted reward
        end

        GRPO->>GRPO: Calculate group relative advantages
        GRPO->>Model: Update policy via gradient descent
        Note over GRPO,Model: KL penalty prevents divergence from base model
    end

    GRPO->>Env: Save checkpoint
```

## Data Flow Architecture

```mermaid
flowchart TD
    subgraph Input
        YAML[Configuration YAML]
        TASK[Task Specification]
    end

    subgraph Generation
        YAML --> LOAD[Config Loader]
        LOAD --> CREATE[Create Dataset]
        TASK --> CREATE
        CREATE --> PROC[Procedural Generation<br/>reasoning_gym]
        PROC --> FMT[Format Questions<br/>Add XML instructions]
    end

    subgraph Training
        FMT --> PROMPT[Training Prompts]
        PROMPT --> LLM[Language Model]
        LLM --> COMP[Completions]
        COMP --> PARSE[Extract Answers<br/>XML Parser]
        PARSE --> VER[Verify Correctness<br/>Task Verifier]
        VER --> REW[Reward Computation]
        REW --> GRAD[GRPO Gradient]
        GRAD --> LLM
    end

    subgraph Output
        LLM --> SAVE[Save Checkpoint]
        SAVE --> EVAL[Evaluation]
        EVAL --> RESULTS[Metrics & Charts]
    end

    style YAML fill:#e1f5ff
    style PROC fill:#ffe1e1
    style LLM fill:#d4edda
    style REW fill:#fff3cd
    style RESULTS fill:#e7d4f7
```

## Module Dependency Graph

```mermaid
graph TD
    subgraph Core Modules
        DU[data/dataset_utils.py]
        DL[data/dataset_loader.py]
        ML[models/model_loader.py]
        TR[training/trainer.py]

        DL --> DU
        TR --> DL
        TR --> ML
    end

    subgraph Reward System
        BR[rewards/base_reward.py]
        CR[rewards/correctness.py]
        RR[rewards/registry.py]

        CR --> BR
        RR --> BR
        RR --> CR
        TR --> RR
    end

    subgraph Utilities
        CFG[utils/config.py]
        MEM[utils/memory_tracker.py]

        TR --> CFG
        TR --> MEM
    end

    subgraph Scripts
        TRAIN[scripts/train.py]
        BENCH[scripts/benchmark_models_parallel.py]
        PLOT[scripts/plot_benchmark.py]

        TRAIN --> TR
        BENCH --> DU
        BENCH --> ML
        PLOT --> BENCH
    end

    subgraph External Dependencies
        RG[reasoning_gym]
        VERIF[verifiers/TRL]
        HF[transformers]
        PEFT[peft]

        DU --> RG
        TR --> VERIF
        ML --> HF
        ML --> PEFT
        CR --> RG
    end

    style DU fill:#ffcccc
    style ML fill:#d4edda
    style TR fill:#fff3cd
    style RR fill:#ffd9b3
    style TRAIN fill:#e1f5ff
    style RG fill:#ffe1e1
```

## Key Architecture Principles

### 1. Unified Dataset Handling
Both training and evaluation use identical functions from `src/data/dataset_utils.py`:
- `format_question()` - Adds XML answer tags
- `extract_answer_from_response()` - Parses model outputs
- `score_answer()` - Verifies correctness using task verifiers

This ensures consistency between training and evaluation.

### 2. Verifier-Driven Learning
The architecture leverages procedurally-generated tasks with built-in verification:
```
Question → Model → Answer → Verifier → Reward → Policy Update
```

This enables scalable training without human labeling.

### 3. Config-Based Design
All experiments are defined through YAML configs with inheritance:
```
base.yaml + model.yaml + task.yaml + experiment.yaml → Training Run
```

No code changes needed for new tasks, models, or hyperparameters.

### 4. Modular Reward System
Registry pattern enables composable rewards:
```python
rewards:
  - name: "correctness"
    weight: 0.8
  - name: "brevity"
    weight: 0.2
```

### 5. Parameter-Efficient Training
LoRA (Low-Rank Adaptation) enables fine-tuning large models:
```
Base Model (frozen) + LoRA Adapters (trainable) = Efficient Fine-tuning
```

## Training Pipeline Flow

1. **Configuration Loading**
   - Load YAML config with inheritance
   - Deep merge base + model + task + experiment configs

2. **Data Preparation**
   - Generate reasoning tasks via reasoning_gym
   - Format questions with XML answer instructions
   - Create HuggingFace Dataset

3. **Reward Setup**
   - Initialize reward functions from registry
   - Create reward rubric with weights
   - Connect to task verifiers

4. **Environment Creation**
   - Wrap dataset + rubric in SingleTurnEnv
   - Define state space (questions) and action space (completions)

5. **Model Loading**
   - Load base LLM from HuggingFace
   - Apply LoRA configuration if enabled
   - Configure tokenizer and special tokens

6. **Training Execution**
   - GRPO loop: generate → extract → verify → reward → update
   - Save checkpoints at specified intervals
   - Log metrics to WandB (optional)

7. **Evaluation**
   - Parallel benchmarking across multiple GPUs
   - Test on held-out tasks
   - Generate accuracy metrics and visualizations

## Evaluation Pipeline Flow

1. **Dataset Creation**
   - Generate fixed test sets (same seed for reproducibility)
   - Use identical formatting as training

2. **Model Loading**
   - Support for HuggingFace models and local checkpoints
   - Automatic LoRA adapter detection

3. **Parallel Execution**
   - Create all (task, model) combinations
   - Round-robin GPU assignment
   - Concurrent evaluation

4. **Scoring**
   - Generate completions with temperature sampling
   - Extract answers using XML parser
   - Verify with task verifiers
   - Compute accuracy metrics

5. **Result Aggregation**
   - Per-task JSON results
   - Multi-task summary statistics
   - Comparison visualizations

## File Structure

```
reasoning-gym/
├── configs/                          # Configuration files
│   ├── base.yaml                     # Default hyperparameters
│   ├── models/                       # Model-specific configs
│   ├── tasks/                        # Task-specific configs
│   └── experiments/                  # Experiment compositions
│
├── src/                              # Core library
│   ├── data/                         # Dataset handling
│   │   ├── dataset_utils.py          # Unified utilities
│   │   └── dataset_loader.py         # Dataset creation
│   ├── models/                       # Model loading
│   │   └── model_loader.py           # HF + LoRA loading
│   ├── rewards/                      # Reward functions
│   │   ├── base_reward.py            # Abstract base
│   │   ├── correctness.py            # Correctness reward
│   │   └── registry.py               # Reward factory
│   ├── training/                     # Training logic
│   │   └── trainer.py                # GRPO orchestration
│   └── utils/                        # Utilities
│       ├── config.py                 # Config loading
│       └── memory_tracker.py         # Memory profiling
│
├── scripts/                          # Executable scripts
│   ├── train.py                      # Training entry point
│   ├── benchmark_models_parallel.py  # Evaluation system
│   └── plot_benchmark.py             # Visualization
│
├── experiments/                      # Training outputs
└── benchmark_results/                # Evaluation results
```

## External Dependencies

- **reasoning_gym**: Procedurally-generated reasoning tasks with verification
- **verifiers/TRL**: GRPO trainer implementation
- **transformers**: HuggingFace model loading and tokenization
- **peft**: Parameter-efficient fine-tuning (LoRA)
- **datasets**: HuggingFace datasets library
- **torch**: PyTorch deep learning framework
- **wandb**: Experiment tracking (optional)

## Technical Innovations

1. **Universal Answer Format**: XML tags work across all task types
2. **Automatic Verification**: Built-in verifiers enable scalable RL training
3. **Unified Pipeline**: Same code for training and evaluation
4. **Composable Rewards**: Registry pattern for extensible reward functions
5. **Parallel Evaluation**: Multi-GPU benchmarking of multiple models
6. **Config Inheritance**: Hierarchical configuration system
7. **Memory Optimization**: LoRA + gradient checkpointing for large models
