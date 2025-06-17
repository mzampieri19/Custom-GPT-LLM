# Custom GPT Language Model Training

## Project Summary

This project demonstrates how to build, train, and use a custom Generative Pre-trained Transformer (GPT) language model from scratch using PyTorch. Inspired by the freeCodeCamp.org tutorial ["Create a Large Language Model from Scratch with Python – Tutorial"](https://www.youtube.com/watch?v=UU1WVnMk4E8&t=6421s), the notebook walks through all steps required to preprocess data, define the model architecture, train the model, and generate new text samples. The project is designed to run efficiently on Apple Silicon (M1/M2/M3) using the Metal backend (`mps`).

---

## Model Versions

1. **model-01**
  - Trained for a few thousand epochs on a minimal dataset and vocabulary.
  - Served as a proof of concept to validate the model architecture.
  - Output was mostly incoherent, but demonstrated basic grammar and sentence structure.

2. **model-02**
  - Used the same dataset as model-01 but trained for approximately 50,000 epochs.
  - Produced text with noticeably improved grammar and structure.
  - Still generated many non-existent words and some incoherent sentences.

3. **model-03**
  - Trained on a much larger dataset (~300,000 lines) of classic literature from Project Gutenberg.
  - Generated text with strong grammar and human-like sentence structure.
  - Occasionally produced non-existent words, but overall output was much closer to real English.

4. **model-04**
  - Utilized a Common Crawl dataset with 1 million lines (~4GB).
  - Training required splitting the dataset into sections due to its size; completed 5,000 epochs so far.
  - Model evaluation is ongoing.

---

## Project Structure

| File Name                  | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `gpt-v1.ipynb`             | Main Jupyter notebook containing all code for data prep, model, training, etc. |
| `training.py`              | A Python script to run the training code                                    |
| `chatbot.py`               | A simple terminal based application of the LLM                              |
| `data_extract.py`          | Script to extract workable data from the JSON file                          |
| `vocab.txt`                | Text file containing the vocabulary (unique characters) for encoding/decoding. |
| `extracted_train_data.txt` | Training data text file.                                                    |
| `extracted_val_data.txt`   | Validation data text file.                                                  |
| `model-01.pkl`             | Saved PyTorch model checkpoint (after training).                            |
| `README.md`                | This documentation file.                                                    |

---

## Step-by-Step Guide

### 1. **Data Preparation**

- **Input Files:**  
  - `extracted_train_data.txt` – The main text corpus for training.
  - `extracted_val_data.txt` – A separate text corpus for validation.

- **Vocabulary Extraction:**  
  - Read all unique characters from your data and save them to `vocab.txt`.
  - This ensures the model can encode/decode every character in your dataset.

### 2. **Encoding and Decoding**

- **Encoding:**  
  - Each character is mapped to a unique integer using dictionaries (`string_to_int` and `int_to_string`).
  - Functions `encode` and `decode` convert between text and integer sequences.

### 3. **Batch Preparation**

- **Random Chunking:**  
  - The function `get_random_chunk(split)` reads a random chunk from the training or validation file.
  - `get_batch(split)` prepares batches of input (`x`) and target (`y`) tensors for training.

### 4. **Model Architecture**

- **Transformer Blocks:**  
  - The model is built from scratch using PyTorch, including:
    - Multi-head self-attention (`Head`, `MultiHeadAttention`)
    - Feed-forward layers (`FeedFoward`)
    - Layer normalization and residual connections (`Block`)
  - The main model class is `GPTLanguageModel`.

### 5. **Training**

- **Device Selection:**  
  - Uses Apple Silicon GPU via `torch.device("mps" if torch.backends.mps.is_available() else "cpu")`.

- **Training Loop:**  
  - Runs for a specified number of iterations (`max_iters`).
  - Periodically evaluates and prints training and validation loss.
  - Uses AdamW optimizer and a learning rate scheduler.

- **Loss Estimation:**  
  - The `estimate_loss()` function computes average loss over several batches for both train and validation splits.

### 6. **Saving and Loading the Model**

- **Saving:**  
  - After training, the model is saved to `model-01.pkl` using Python's `pickle` module.

- **Loading:**  
  - The model can be reloaded from the pickle file for further training or inference.

### 7. **Text Generation**

- **Sampling:**  
  - The trained model can generate new text by sampling from the learned probability distribution.
  - The `generate()` method in the model class handles this.

---

## How to Replicate This Project

1. **Clone or Download the Repository**

2. **Prepare Your Data**
   - Place your training and validation text files as `extracted_train_data.txt` and `extracted_val_data.txt`.
   - Ensure `vocab.txt` contains all unique characters from your data.

3. **Set Up the Python Environment**
   - Use Python 3.9+ (recommended).
   - Install dependencies:
     ```
     pip install torch torchvision
     ```
     For Apple Silicon, use the nightly build for best MPS support:
     ```
     pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
     ```

4. **Open `gpt-v1.ipynb` in VS Code or Jupyter**
   - Step through each cell, making sure paths to data files are correct.

5. **Train the Model**
   - Run the training loop cell.
   - Monitor training and validation loss.

6. **Save and Reload the Model**
   - After training, the model is saved automatically.
   - You can reload it for further training or text generation.

7. **Generate Text**
   - Use the provided code to generate new text samples from the trained model.

---

## Tips for Best Results

- **Data Quality:** Clean and preprocess your data for best results.
- **Hyperparameters:** Tune `n_embd`, `n_head`, `n_layer`, `learning_rate`, and `dropout` for your dataset.
- **Compute:** Larger models and more data require more compute and memory.
- **Validation:** Always monitor validation loss to avoid overfitting.

---

## Credits

- Based on the freeCodeCamp.org tutorial. 
- Custom modifications and training by Michelangelo Zampieri.

---