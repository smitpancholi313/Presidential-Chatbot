import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample, compute_class_weight
import torch
from sklearn.metrics import classification_report
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

nltk.download('punkt')

spacy_model = spacy.load('en_core_web_sm')

us_speeches_path = "speeches.xlsx"
russian_speeches_path = "speeches_russian_PM.xlsx"

us_speeches = pd.read_excel(us_speeches_path)
russian_speeches = pd.read_excel(russian_speeches_path)


def remove_president_name(text):
    if isinstance(text, str) and ":" in text:
        return text.split(":", 1)[1].strip()
    return text


russian_speeches['transcript'] = russian_speeches['transcript_unfiltered'].apply(remove_president_name)

russian_speeches = russian_speeches.drop(columns=['transcript_unfiltered', 'transcript_filtered', 'teaser', 'tags'])
russian_speeches = russian_speeches.loc[:, ~russian_speeches.columns.str.contains('^Unnamed')]
russian_speeches = russian_speeches.dropna(axis=1, how='all')

columns = list(russian_speeches.columns)
columns.insert(columns.index('President') + 1, columns.pop(columns.index('transcript')))
russian_speeches = russian_speeches[columns]

presidents = ['Vladimir Putin', 'Boris Yeltsin', 'Dmitry Medvedev']


def normalize_president_name(name):
    if not isinstance(name, str):
        return None
    name_lower = name.lower()
    for valid_name in presidents:
        if valid_name.lower() in name_lower:
            return valid_name
    return None


russian_speeches['President'] = russian_speeches['President'].apply(normalize_president_name)
russian_speeches = russian_speeches.dropna(subset=['President'])

class_counts = russian_speeches['President'].value_counts()
sufficient_classes = class_counts[class_counts >= 2].index
russian_speeches = russian_speeches[russian_speeches['President'].isin(sufficient_classes)]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_texts_with_transformers(texts):
    tokenized_texts = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return tokenized_texts.input_ids


def preprocess_text(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]
    return " ".join(tokens)


us_speeches['transcript'] = us_speeches['transcript'].apply(preprocess_text)
russian_speeches['transcript'] = russian_speeches['transcript'].apply(preprocess_text)

from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    russian_speeches['transcript'], russian_speeches['President'], test_size=0.2, random_state=42,
    stratify=russian_speeches['President']
)

train_data = pd.DataFrame({"texts": train_texts, "labels": train_labels})
majority = train_data[train_data['labels'] == 'Vladimir Putin']
minorities = train_data[train_data['labels'] != 'Vladimir Putin']
minorities_oversampled = resample(minorities, replace=True, n_samples=len(majority), random_state=42)
balanced_train_data = pd.concat([majority, minorities_oversampled])
balanced_train_texts = balanced_train_data['texts'].tolist()
balanced_train_labels = balanced_train_data['labels'].tolist()

label_mapping = {name: idx for idx, name in enumerate(presidents)}
balanced_train_labels = [label_mapping[label] for label in balanced_train_labels]
val_labels = val_labels.map(label_mapping).astype(int)


class SpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


train_encodings = tokenizer(balanced_train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512, return_tensors="pt")

train_dataset = SpeechDataset(train_encodings, balanced_train_labels)
val_dataset = SpeechDataset(val_encodings, val_labels.tolist())

present_classes = np.unique(balanced_train_labels)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=present_classes,
    y=np.array(balanced_train_labels)
)

full_class_weights = np.zeros(len(presidents))
for idx, weight in zip(present_classes, class_weights_array):
    full_class_weights[idx] = weight

class_weights = torch.tensor(full_class_weights, dtype=torch.float).to(device)

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch.nn as nn


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(presidents))

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

predictions = trainer.predict(val_dataset)
logits = predictions.predictions
predicted_labels = np.argmax(logits, axis=1)
subset_presidents = [presidents[i] for i in np.unique(val_labels)]
print("Classification Report:")
print(classification_report(val_labels.tolist(), predicted_labels, target_names=subset_presidents))

test_text = "Dear Russians, In a few hours we will see a magical date on our calendars, the year 2000, a new century, a new millennium."
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
inputs = {key: val.to(device) for key, val in inputs.items()}
model.to(device)
outputs = model(**inputs)
predicted_class = outputs.logits.argmax().item()
print("\nTesting Custom Input Text:", test_text)
print("Predicted President:", presidents[predicted_class])
