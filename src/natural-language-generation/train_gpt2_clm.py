from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from datasets import load_dataset
import json
from pathlib import Path
import random
import logging
import math
import torch
import os

# 0) SETUP VARS AND LOGGING
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SEED = 10
random.seed(SEED)  # for reproducibility

# 1) LOAD THE DATASET
# Select relevant fields from original format and merge them in plain txt file
# TODO: wrap this with a preprocessing function
dataset_dir = Path(f"{SCRIPT_DIR}/../../datasets/recipes_raw/")  # Use Path class for easy iteration over files
dataset = []
for file in dataset_dir.iterdir():
    if file.name.endswith('.json'):
        with open(file) as fn:
            data = json.load(fn)
        for id in data.keys():
            try:  # Some elements are empty and throw "KeyError" exception
                title = data[id]['title']
                ingredients = ' '.join([ing for ing in data[id]['ingredients']])
                instructions = data[id]['instructions']
                sentence = f"{title}, {ingredients}, {instructions}"
                dataset.append(sentence)
            except KeyError:
                continue
logger.info(f"Total number of sentences: {len(dataset)}")

# Clean data
# TODO: improve text cleaning by removing multiple spaces, and multiple commas
# TODO: create a clean function
dataset = [sentence.replace('\n', ' ').replace('ADVERTISEMENT', ',') for sentence in dataset]

# Create train and dev splits
random.shuffle(dataset)  # shuffle sentences first
max_size = 100  # set for fast debugging
dataset = dataset[:max_size]
dev_size = round(0.1 * len(dataset))
dev_data = dataset[:dev_size]
train_data = dataset[dev_size:]

# Write to file
for split, data in zip(["train", "dev"], [train_data, dev_data]):
    with open(f"{split}.txt", 'w') as fn:
        fn.writelines("\n".join(data))

logger.info(f"Created splits of size,"
            f"train.json: {len(train_data)}, "
            f"dev.json: {dev_size}")

# Use the "load_dataset" method with the "json" builder to create the features
dataset = load_dataset('text', data_files={'train': 'train.txt',
                                           'dev': 'dev.txt'})

# 2) TOKENIZE DATA AND PREPARE INPUTS AND LABELS
# Load tokenizer
model_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# TODO: add PAD and <|startofsentence|> special tokens to the GPT-2 model for data padding and sentence marking,
#  and learn them during training. The PAD tokens allow the construction a variable-size batches and avoid the
#  need for a `group_texts` function
# Append the <|endoftext|> special token at the end of each sentence
def tokenize(sentences):
    return tokenizer([sentence + tokenizer.eos_token for sentence in sentences['text']])


# Taken from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
# Group sentences in batches of equal size to avoid padding
def group_texts(examples):
    # Concatenate all texts.
    block_size = 1024  # set the "blocks" to the maximum GPT-2 model length
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


logger.info("Prepare data...")
tokenized_dataset = dataset.map(tokenize,
                                batched=True,
                                remove_columns=['text'],  # remove text feature we do not need anymore
                                desc='Tokenize train and dev datasets')

clm_dataset = tokenized_dataset.map(group_texts,
                                    batched=True,
                                    desc='Group text for language model training')

# 3) TRAIN THE CAUSAL LANGUAGE MODEL
# We use the "Trainer" API to perform the training loop
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             config=config)

training_args = TrainingArguments(no_cuda=bool(torch.cuda.is_available()),
                                  per_device_train_batch_size=1,
                                  per_device_eval_batch_size=1,
                                  gradient_accumulation_steps=4,
                                  max_steps=2,
                                  logging_steps=1,
                                  output_dir='gpt2-recipes')

# TODO: use a padding data collator if the PAD token is added (remember to update the model embeddings with
#  `model.resize_token_embeddings(len(tokenizer))`
trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=clm_dataset['train'],
                  eval_dataset=clm_dataset['dev'],
                  tokenizer=tokenizer,
                  # Since GPT2 do not use PAD special tokens, we use the default data collator
                  data_collator=default_data_collator)

logger.info('Training...')
train_result = trainer.train()
trainer.save_model('gpt2-recipes')

# Save the metrics (loss on the training data in our case)
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# 4) EVALUATE ON DEV SET WITH PERPLEXITY
metrics = trainer.evaluate()

# compute perplexity
try:
    perplexity = math.exp(metrics["eval_loss"])
except OverflowError:
    perplexity = float("inf")
metrics["perplexity"] = perplexity

# Save the metrics (loss on the training data in our case)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
