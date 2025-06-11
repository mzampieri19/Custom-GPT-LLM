import os
import json
import tqdm
import random

input_file = 'openwebtext-10k.jsonl'
output_train_file = 'extracted_train_data.txt'
output_val_file = 'extracted_val_data.txt'
vocab_file = 'vocab.txt'

vocab = set()
train_ratio = 0.9  # 90% train, 10% val

# Read all lines from the input file
with open(input_file, 'r') as f:
    lines = f.readlines()

random.shuffle(lines)  # Shuffle for random split

split_idx = int(len(lines) * train_ratio)
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

def process_and_write(lines, output_file):
    with open(output_file, 'w') as out_f:
        for line in tqdm.tqdm(lines):
            data = json.loads(line)
            text = data.get('text', '')
            if text:
                out_f.write(text + '\n')
                words = text.split()
                vocab.update(words)

process_and_write(train_lines, output_train_file)
process_and_write(val_lines, output_val_file)

# Write the vocabulary to a file
with open(vocab_file, 'w') as vocab_f:
    for word in sorted(vocab):
        vocab_f.write(word + '\n')
