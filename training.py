import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm

# ------------------------------------------------------
# 1. Configuration & Hyperparams
# ------------------------------------------------------

DATA_DIR = "data_processed"
SECTION_TYPES = ["Summary", "Param", "Return"]
SPLITS = ["train", "valid", "test"]

MAX_LEN_CODE = 64
MAX_LEN_COMMENT = 256

BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
MODEL_NAME = "microsoft/codebert-base"

# Option to freeze embeddings
FREEZE_EMBEDDINGS = True
FREEZE_ENCODER_LAYERS = 0


# ------------------------------------------------------
# 2. Custom Dataset
# ------------------------------------------------------
class JavadocCodeDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 section_type,
                 tokenizer,
                 max_length_code=64,
                 max_length_comment=256):
        """
        data_dir: base directory for that section type and split
        split: one of 'train', 'valid', 'test'
        section_type: 'Summary', 'Param', 'Return'
        tokenizer: RobertaTokenizer
        max_length_code, max_length_comment: used to safely truncate if desired
                                             (but we combine them into one).
        """
        self.data_dir = os.path.join(data_dir, section_type, split)
        self.tokenizer = tokenizer
        self.max_length_code = max_length_code
        self.max_length_comment = max_length_comment
        self.samples = []

        data_path = os.path.join(self.data_dir, "data.json")
        with open(data_path, "r") as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples[0])

    def __getitem__(self, idx):
        all_samples = self.samples[0]
        sample = all_samples[idx]

        label_val = sample["label"]
        if label_val == 0:
            # consistent
            comment_tokens = sample["new_comment_subtokens"]
            code_tokens = sample["new_code_subtokens"]
        else:
            # inconsistent
            comment_tokens = sample["old_comment_subtokens"]
            code_tokens = sample["new_code_subtokens"]

        comment_str = " ".join(comment_tokens)
        code_str = " ".join(code_tokens)

        # Combine comment + code into one text with a [SEP] token in between
        # "[COMMENT] [SEP] [CODE]"
        combined_input = f"{comment_str} {self.tokenizer.sep_token} {code_str}"

        max_len_total = self.max_length_code + self.max_length_comment

        encoded = self.tokenizer(
            combined_input,
            truncation=True,
            padding="max_length",
            max_length=max_len_total,
            return_tensors="pt"
        )

        label_tensor = torch.tensor(label_val, dtype=torch.float)

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": label_tensor
        }


# ------------------------------------------------------
# 3. Dataloader Creation
# ------------------------------------------------------
def create_dataloaders(data_dir,
                       section_types,
                       tokenizer,
                       batch_size=8,
                       max_length_code=64,
                       max_length_comment=256):
    dataloaders = {}
    for section_type in section_types:
        ds_dict = {}
        for split in SPLITS:
            ds = JavadocCodeDataset(
                data_dir=data_dir,
                split=split,
                section_type=section_type,
                tokenizer=tokenizer,
                max_length_code=max_length_code,
                max_length_comment=max_length_comment
            )
            dl = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
            ds_dict[split] = dl
        dataloaders[section_type] = ds_dict
    return dataloaders


tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
dataloaders = create_dataloaders(
    DATA_DIR,
    SECTION_TYPES,
    tokenizer,
    BATCH_SIZE,
    MAX_LEN_CODE,
    MAX_LEN_COMMENT
)


# ------------------------------------------------------
# 4. Freeze Helper
#    Freeze embeddings and possibly first N encoder layers if desired
# ------------------------------------------------------
def freeze_embeddings_and_layers(model, freeze_embeddings=True, freeze_encoder_layers=0):
    """
    model: RobertaForSequenceClassification
    freeze_embeddings: bool
    freeze_encoder_layers: int -> number of initial layers to freeze
    """
    if freeze_embeddings:
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False

    # Freeze first N layers in the encoder
    for layer_idx in range(freeze_encoder_layers):
        for param in model.roberta.encoder.layer[layer_idx].parameters():
            param.requires_grad = False


# ------------------------------------------------------
# 5. Training & Evaluation Functions
# ------------------------------------------------------
def train_section(model,
                  dataloaders,
                  section_type,
                  optimizer,
                  criterion,
                  scheduler=None,
                  num_epochs=5,
                  device=DEVICE,
                  track_metric="f1",
                  early_stopping_patience=2,
                  save_checkpoint=False):
    """
    model: RobertaForSequenceClassification (num_labels=1)
    track_metric: 'loss' or 'f1'
    """
    best_val_metric = 0.0 if track_metric == "f1" else float('inf')
    patience_counter = 0

    if save_checkpoint:
        checkpoint_dir = f"./checkpoints_{section_type}"
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(dataloaders[section_type]['train'], desc=f"Training {section_type} epoch {epoch + 1}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits.squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(dataloaders[section_type]['train'])

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in dataloaders[section_type]['valid']:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits.squeeze(-1), labels)
                total_val_loss += loss.item()

                probs = torch.sigmoid(logits.squeeze(-1))
                val_preds.extend((probs > 0.5).int().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(dataloaders[section_type]['valid'])
        val_acc = accuracy_score(val_labels, val_preds)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='binary', zero_division=0
        )

        print(
            f"[{section_type} | Epoch {epoch + 1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Valid Loss: {avg_val_loss:.4f}, "
            f"Valid Acc: {val_acc:.4f}, "
            f"Valid Prec: {val_prec:.4f}, "
            f"Valid Rec: {val_rec:.4f}, "
            f"Valid F1: {val_f1:.4f}"
        )

        if save_checkpoint:
            ckpt_path = os.path.join(checkpoint_dir, f"{section_type}_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)

        # Early Stopping
        if track_metric == "loss":
            current_metric = avg_val_loss
            is_improvement = current_metric < best_val_metric
        else:  # track_metric == "f1"
            current_metric = val_f1
            is_improvement = current_metric > best_val_metric

        if is_improvement:
            patience_counter = 0
            best_val_metric = current_metric
            torch.save(model.state_dict(), f"{section_type}_best_model.pt")
        else:
            patience_counter += 1
            if patience_counter > early_stopping_patience:
                print(f"Early stopping triggered for {section_type}")
                break


def evaluate_section(model, dataloader, section_type, device=DEVICE):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_vals = batch["label"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)  # shape [B]
            probs = torch.sigmoid(logits)

            pred = (probs > 0.5).int().cpu().numpy()
            preds.extend(pred)
            labels.extend(label_vals)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return acc, prec, rec, f1


# ------------------------------------------------------
# 6. Main Training & Evaluation
# ------------------------------------------------------
def main():
    criterion = nn.BCEWithLogitsLoss()

    for section_type in SECTION_TYPES:
        print(f"--- Training classifier for {section_type} section ---")

        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1
        ).to(DEVICE)

        freeze_embeddings_and_layers(
            model,
            freeze_embeddings=FREEZE_EMBEDDINGS,
            freeze_encoder_layers=FREEZE_ENCODER_LAYERS
        )

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        num_training_batches = len(dataloaders[section_type]["train"])
        total_steps = NUM_EPOCHS * num_training_batches
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        train_section(
            model=model,
            dataloaders=dataloaders,
            section_type=section_type,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            track_metric="f1",
            save_checkpoint=False
        )

        model.load_state_dict(torch.load(f"{section_type}_best_model.pt"))

        test_acc, test_prec, test_rec, test_f1 = evaluate_section(
            model, dataloaders[section_type]['test'], section_type, DEVICE
        )
        print(
            f"[TEST: {section_type}] "
            f"Acc={test_acc:.4f}, Prec={test_prec:.4f}, "
            f"Rec={test_rec:.4f}, F1={test_f1:.4f}"
        )


def test_models():
    MODEL_DIR = "default_head_models"
    for section_type in SECTION_TYPES:
        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1
        ).to(DEVICE)

        model.load_state_dict(torch.load(f"{MODEL_DIR}/{section_type}_best_model.pt"))

        test_acc, test_prec, test_rec, test_f1 = evaluate_section(
            model, dataloaders[section_type]['test'], section_type, DEVICE
        )
        print(
            f"[TEST: {section_type}] "
            f"Acc={test_acc:.4f}, Prec={test_prec:.4f}, "
            f"Rec={test_rec:.4f}, F1={test_f1:.4f}"
        )


if __name__ == "__main__":
    main()
