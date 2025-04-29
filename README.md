# Fine-tuning DeepSeek-R1-Distill-Qwen-1.5B with LoRA on SmolTalk Dataset

This repository contains a Jupyter Notebook (`.ipynb`) demonstrating how to fine-tune the `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` language model for improved conversational abilities using the `HuggingFaceTB/smoltalk` dataset. The fine-tuning process utilizes Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation) and optional 4-bit quantization for efficient training on consumer hardware.

## Overview

The goal of this project is to adapt the base DeepSeek model to better handle everyday chit-chat and small talk scenarios present in the SmolTalk dataset. Key techniques and libraries used include:

*   **Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` - A distilled version of DeepSeek, potentially offering a good balance between performance and resource requirements.
*   **Dataset:** `HuggingFaceTB/smoltalk` (specifically the `everyday-conversations` subset) - A dataset containing conversational exchanges.
*   **Fine-tuning Technique:** LoRA (Low-Rank Adaptation) via the `peft` library. This significantly reduces the number of trainable parameters, making fine-tuning faster and less memory-intensive.
*   **Quantization (Optional):** 4-bit quantization using `bitsandbytes` to further reduce memory footprint during training and inference.
*   **Framework:** Hugging Face `transformers` library for model loading, tokenization, training (`Trainer`), and data processing (`datasets`).

## Features

*   Fine-tunes `DeepSeek-R1-Distill-Qwen-1.5B`.
*   Uses the `smoltalk` dataset for conversational tuning.
*   Implements LoRA for efficient training.
*   Includes 4-bit quantization support (`BitsAndBytesConfig`).
*   Preprocesses data using the model's chat template.
*   Provides training script using `transformers.Trainer`.
*   Includes code for testing the fine-tuned model with an interactive chat loop.
*   Shows how to merge the LoRA adapter with the base model.

## Prerequisites

*   **Python 3.8+**
*   **PyTorch:** (`torch`) Install matching your CUDA version.
*   **CUDA-enabled GPU:** Required for training and efficient inference.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```

## Usage

The primary code is located in the Jupyter Notebook (`.ipynb`) file provided in the repository.

1.  **Launch Jupyter:**
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
    Open the `.ipynb` file in your browser.

2.  **Configuration (Optional):**
    Before running the cells, you can adjust parameters in the `### Configuration` section of the notebook:
    *   `model_id`: Change if you want to try a different base model (ensure compatibility).
    *   `dataset_id`: Change the dataset source if needed.
    *   `output_dir`: Directory where the trained LoRA adapter and logs will be saved.
    *   `rank_dimension`, `lora_alpha`, `lora_dropout`: LoRA specific parameters. Experimenting here can affect performance and training stability.
    *   `bnb_config`: Modify or remove this to change quantization settings (e.g., disable 4-bit loading).
    *   `TrainingArguments`: Adjust hyperparameters like `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `num_train_epochs`, etc., based on your hardware and desired training outcome.

3.  **Run the Notebook Cells:**
    Execute the cells sequentially.
    *   **Setup & Loading:** Loads the tokenizer, base model (with quantization), and dataset.
    *   **Preprocessing:** Formats the conversations using the chat template and tokenizes them.
    *   **LoRA Setup:** Applies the LoRA configuration to the model.
    *   **Training:** Initializes the `Trainer` and starts the fine-tuning process. This will take time depending on your hardware and configuration. The LoRA adapter checkpoints will be saved periodically (based on `save_strategy`) and the final adapter will be saved to `output_dir`.
    *   **Testing:** After training, this section loads the trained LoRA adapter, merges it into the base model (optional but done here for standalone inference), and provides an interactive chat interface to test the fine-tuned model's responses.

4.  **Inference with the Fine-tuned Model:**
    *   The notebook saves the final LoRA adapter to the specified `output_dir` (e.g., `./smoltalk-finetuned-deepseek-lora`).
    *   It also demonstrates merging the adapter with the base model and saving the complete merged model to `./smoltalk-finetuned-merged`.
    *   You can use the code in the "Test Model" section (or adapt it) to load either the base model + adapter or the merged model for generating responses.

## Code Explanation

*   **Libraries & Configuration:** Imports necessary libraries and sets up key parameters like model/dataset IDs, LoRA config, quantization config, and output directory.
*   **Load Tokenizer and Model:** Loads the specified model and tokenizer from Hugging Face Hub. Applies 4-bit quantization using `BitsAndBytesConfig`. Handles potential missing padding tokens.
*   **Load and Prepare Dataset:** Loads the `smoltalk` dataset, splits it into training and evaluation sets, and defines a `preprocess_conversations` function. This function applies the model's specific chat template (`tokenizer.apply_chat_template`) to format conversations correctly before tokenization.
*   **Apply PEFT (LoRA):** Prepares the model for k-bit training (if quantized) and applies the `LoraConfig` using `get_peft_model`. Prints the number of trainable parameters.
*   **Define Training Arguments:** Sets up hyperparameters for training (batch size, learning rate, epochs, saving strategy, optimizer, etc.) using `TrainingArguments`. Enables mixed-precision training (BF16/FP16) if supported.
*   **Define Data Collator:** Uses `DataCollatorForLanguageModeling` to handle dynamic padding within batches.
*   **Initialize Trainer:** Creates a `Trainer` instance with the model, arguments, datasets, tokenizer, and data collator.
*   **Start Fine-tuning:** Calls `trainer.train()` to begin the training loop. Saves the final adapter using `trainer.save_model()`.
*   **Test Model:** Loads the trained LoRA adapter, merges it with the base model, sets the model to evaluation mode, defines a `generate_response` function for inference, and provides example usage including an interactive loop.
