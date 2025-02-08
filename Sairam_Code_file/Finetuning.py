#%%
import os
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, pipeline
from torch.utils.data import DataLoader, Dataset

# Using cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

#%%
# Hardcoded path
filepath = "/home/ubuntu/sairam/Project/speeches.xlsx"
data = pd.read_excel(filepath)
print(data.head())

#%%
print(data['transcript'])

# %% Prepping specch data for finetuning
class SpeechDataset(Dataset):
    def __init__(self, transcripts, tokenizer, max_length=512):
        self.transcripts = transcripts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, idx):
        transcript = self.transcripts[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            transcript,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}


# %%
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Adding special token for context seperation.
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

# Prepare Dataset
transcripts = data['transcript'].tolist()
dataset = SpeechDataset(transcripts, tokenizer)

from torch.utils.data import DataLoader

batch_size = 4  # Small batch size for memory efficiency
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#%% Loading GPT-2 Model

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))  # Adjust for any added tokens

model = model.to(device)

#%% Training loop code
optimizer = AdamW(model.parameters(), lr=5e-5)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
#%%
# It seems the loss stagnates after 10 epochs or so
epochs = 20
total_steps = len(dataloader) * epochs

model.train()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # print("Input_ids are ",input_ids)
        # print("attention_mask is ",attention_mask)
        # print("labels is ",labels)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f}")

#%% Saving the model ina directory on the instance, for testing
output_dir = "/home/ubuntu/sairam/Project/test_model"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")

#%%
#%%

# Testing the fine tuned model whether it has actually learnt or not
from transformers import pipeline

fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

text_generator = pipeline(
    "text-generation",
    model=fine_tuned_model,
    tokenizer=fine_tuned_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

print("Ask a question to the President (type 'exit' to quit):")

while True:
    input_text = input("You: ")
    if input_text.lower() == "exit":
        print("Exiting... Goodbye!")
        break

    # Generate response
    generated_text = text_generator(input_text, max_length=300, num_return_sequences=1)
    print(f"President: {generated_text[0]['generated_text']}")
