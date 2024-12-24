import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ------------------------------------------------------
# 1. Configuration & Hyperparams
# ------------------------------------------------------

DATA_DIR = "data_processed"
SECTION_TYPES = ["Summary", "Param", "Return"]
SPLITS = ["train", "valid", "test"]

MAX_LEN_CODE = 64
MAX_LEN_COMMENT = 256
BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
MODEL_NAME = "microsoft/codebert-base"


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


def evaluate_section(model, dataloader, section_type, device=DEVICE):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_vals = batch["label"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            probs = torch.sigmoid(logits)

            pred = (probs > 0.5).int().cpu().numpy()
            preds.extend(pred)
            labels.extend(label_vals)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return acc, prec, rec, f1


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
    test_models()
