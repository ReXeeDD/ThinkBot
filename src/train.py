import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Paths
model_path = "C:/Users/albin/OneDrive/Documents/coder saves/ThinkBot1/model/pytorch_model.bin"
refined_text_path = "C:/Users/albin/OneDrive/Documents/coder saves/ThinkBot1/src/train.txt"
output_model_path = "C:/Users/albin/OneDrive/Documents/coder saves/ThinkBot1/model/refined_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")  # Ensure the correct tokenizer is used
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")  # Ensure the correct model is used
model.load_state_dict(torch.load(model_path), strict=False)  # Allow loading with mismatched keys

# Prepare dataset
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

dataset = load_dataset(refined_text_path, tokenizer)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_model_path,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# Fine-tune the model
trainer.train()

# Save the refined model
trainer.save_model(output_model_path)
tokenizer.save_pretrained(output_model_path)

print(f"Refined model saved to {output_model_path}")