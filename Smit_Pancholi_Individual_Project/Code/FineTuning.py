import os
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

filepath1 = "speeches.xlsx"
filepath2 = "speeches_russian_PM.xlsx"

data1 = pd.read_excel(filepath1)
data2 = pd.read_excel(filepath2)
data2 = data2.rename(columns={"transcript_filtered": "transcript"})

data = pd.concat([data1, data2], ignore_index=True)
data['transcript'] = data['transcript'].fillna("").astype(str)

class SpeechDataset(Dataset):
    def __init__(self, transcripts, tokenizer, max_length=512):
        self.transcripts = transcripts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, idx):
        transcript = self.transcripts[idx]
        encoding = self.tokenizer(
            transcript,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        if input_ids.size(1) != self.max_length:
            print(f"Index {idx} has input_ids shape: {input_ids.size()} - Fixing by padding.")
            input_ids = torch.full((1, self.max_length), self.tokenizer.pad_token_id)
            attention_mask = torch.zeros((1, self.max_length), dtype=torch.long)

        input_ids = input_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})


transcripts = data['transcript'].tolist()
dataset = SpeechDataset(transcripts, tokenizer)

batch_size = 4  
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer)) 
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 5
total_steps = len(dataloader) * epochs

model.train()
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.4f}")

output_dir = "./fine_tuned_president_5_epochs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")
