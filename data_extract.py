import os
import tqdm
import random

input_dir = 'data'
output_train_file = 'train_data.txt'
output_val_file = 'val_data.txt'
vocab_file = 'vocab_books.txt'

vocab = set()
train_ratio = 0.9  # 90% train, 10% val

# Read all lines from all txt files in the input directory
lines = []
for file in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file)
    if os.path.isfile(file_path) and file.endswith('.txt'):
        with open(file_path, 'r') as f:
            lines.extend(f.readlines())

random.shuffle(lines)  # Shuffle for random split

split_idx = int(len(lines) * train_ratio)
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

def process_and_write(lines, output_file):
    with open(output_file, 'w') as out_f:
        for line in tqdm.tqdm(lines):
            text = line.strip()
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
