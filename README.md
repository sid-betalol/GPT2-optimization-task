# GPT-2 Training Script Development

A training script for the GPT-2 model. This endeavor has been an enlightening journey, providing deep insights into the complexities of language model training. 

## Overview

This Python-based training script leverages the robust PyTorch framework alongside Hugging Face's Transformers, aiming to fine-tune the GPT-2 model efficiently. The script's key features include:

- **Model Configuration**: A dataclass is utilized for GPT-2 configurations, allowing for versatile model parameter adjustments.
- **Custom Dataset Integration**: Supports diverse dataset formats, including file-based and Hugging Face datasets.
- **Gradient Checkpointing**: Facilitates memory-efficient training, a critical aspect for managing large models like GPT-2.
- **Optimized Training Loop**: Incorporates the Accelerate library to streamline CPU/GPU and distributed training processes.

## Important Reminder

- **Accelerator Configuration**: Ensure to configure the accelerator by running `accelerate init` in the command line before starting the training process. This step is crucial for optimal hardware utilization.

## Notable Learnings and Decisions

- **AdamW Optimizer**: The adoption of AdamW, an optimizer with a weight decay fix, is pivotal. It provides more effective L2 regularization than traditional Adam, enhancing training stability and model generalization.
- **Learning Rate Scheduler**: A cosine decay learning rate scheduler was implemented, optimizing learning rate adjustments throughout the training cycle for superior model performance.
- **Weight Initialization and Tying**: Includes strategic weight initialization and optional weight tying, which can reduce the model's parameter count and improve performance in language modeling tasks.
- **Reproducibility with Seed Setting**: Ensures consistent and replicable results across different runs by setting a random seed, a vital aspect for credible machine learning experimentation.
- **Gradient Accumulation**: Employs gradient accumulation to manage large batches on limited hardware, enabling effective training with larger batch sizes.

## Challenges and Resolutions

- **Memory Management**: Addressed the high memory demands of GPT-2 through gradient checkpointing and optimal batch sizing, enhancing GPU memory utilization.
- **Hyperparameter Tuning**: Extensive experimentation was conducted to identify the optimal hyperparameter set, focusing on the balance between learning rate, batch size, and epoch count.

## References
- [Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners.](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [nanoGPT | Andrej Karpathy](https://github.com/karpathy/nanoGPT)
- [Let's build GPT: from scratch, in code, spelled out. | Andrej Karpathy on YouTube](https://youtu.be/kCc8FmEb1nY?si=AKphijxUHMc2CUXo)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864.pdf)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245.pdf)
- [Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150v2.pdf)
