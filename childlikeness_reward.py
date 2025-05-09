import sys
import math
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Detected device:", device)

print("Loading dataset...")
data = pd.read_csv('path/to/child_utterances')
data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'avg_aoa', 'age'], inplace=True)
data.rename(columns={'utterance': 'text', 'year': 'label'}, inplace=True)

data = data[(data['label'] > 0) & (data['label'] <= 4)]
print(f"Total examples after filtering: {len(data)}")

# Get binary labels
data['label'] = (data['label'] <= 2).astype(int)

label_counts = data['label'].value_counts()
print(f"\nLabel distribution:\n{label_counts.to_string()}")
label_counts.to_csv("label_distribution.csv", index=True)

print("\nSplitting data into train and test sets...")
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)
print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

def create_input_examples(df):
    return [InputExample(texts=[row['text']], label=row['label']) for _, row in df.iterrows()]

train_examples = create_input_examples(train_data)
test_examples = create_input_examples(test_data)

train_loader = DataLoader(train_examples, shuffle=True, batch_size=1024)
test_loader = DataLoader(test_examples, shuffle=False, batch_size=1024)

evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_examples, show_progress_bar=False)

print("\nInitializing model...")
model = CrossEncoder(
    'google-bert/bert-base-uncased',
    num_labels=1,
    automodel_args={},
    tokenizer_args={"max_length": 32, "padding": "max_length", "truncation": True},
    max_length=32,
    classifier_dropout=0.2
)

# Training
epochs = 8
warmup_steps = math.ceil(len(train_loader) * epochs * 0.05)
eval_steps = math.ceil(len(train_loader)) * 3
print(f"\nTraining for {epochs} epochs")
print(f"Warmup steps: {warmup_steps}, Evaluation every {eval_steps} steps")

def report_score(score, epoch, step):
    print(f"[Callback] Score at epoch {epoch}, step {step}: {score:.4f}")

print("\nStarting training...\n")
model.fit(
    train_dataloader=train_loader,
    evaluator=evaluator,
    epochs=epochs,
    warmup_steps=warmup_steps,
    output_path="binary_results",
    callback=report_score,
    show_progress_bar=True,
    save_best_model=True,
    evaluation_steps=eval_steps
)

print("\nEvaluating model on test set...")
final_score = evaluator(model, test_loader)
print(f"\nFinal evaluation score: {final_score:.4f}")
