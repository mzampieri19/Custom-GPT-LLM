from datasets import load_dataset
from tqdm import tqdm

# Stream only the first N samples from the train split
ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    "sample-10BT",
    split="train",
    streaming=True
)

subset = []
for i, sample in enumerate(tqdm(ds, total=1000000, desc="Streaming samples")):
    if i >= 1000000:  
        break
    subset.append(sample)

# Split into 90% train and 10% val
split_idx = int(0.9 * len(subset))
train_subset = subset[:split_idx]
val_subset = subset[split_idx:]

print(f"Train samples: {len(train_subset)}")
print(f"Val samples: {len(val_subset)}")

# Write train samples
with open('crawl_data_train.txt', 'w', encoding='utf-8') as f:
    for sample in tqdm(train_subset, desc="Writing train samples"):
        f.write(sample['text'] + '\n')

# Write val samples
with open('crawl_data_val.txt', 'w', encoding='utf-8') as f:
    for sample in tqdm(val_subset, desc="Writing val samples"):
        f.write(sample['text'] + '\n')