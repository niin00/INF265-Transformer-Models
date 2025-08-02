import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class QADataset(Dataset):
    def __init__(self, config, tokenizer):
        self.dataset = load_dataset(config.dataset)[config.split]
        n_subset = int(config.model_train_fraction * len(self.dataset))
        self.dataset = self.dataset.select(range(n_subset))
        print(
            f"Loaded dataset of size {len(self.dataset)} with columns {self.dataset.column_names}"
        )

        self.tokenizer = tokenizer
        self.max_length = config.max_len

        # Special token IDs (use these IDs in the __getitem__ method)
        self.pad_id = self.tokenizer.token_to_id(config.pad_token)
        self.sep_id = self.tokenizer.token_to_id(config.sep_token)
        self.end_id = self.tokenizer.token_to_id(config.end_token)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question, answer = self.dataset[idx]["question"], self.dataset[idx]["answer"]

        # Tokenize question and answer
        q_ids = self.tokenizer.encode(question).ids
        a_ids = self.tokenizer.encode(answer).ids
        
        # Concatenate: question [SEP] answer [END]
        full_seq = q_ids + [self.sep_id] + a_ids
        full_seq.append(self.end_id)
        
        # We want a total length of (max_length + 1) to allow shifting for target
        desired_len = self.max_length + 1
        # Truncate if too long
        if len(full_seq) > desired_len:
            full_seq = full_seq[:desired_len]
        # Pad if too short
        if len(full_seq) < desired_len:
            full_seq += [self.pad_id] * (desired_len - len(full_seq))  
            
        # Split into source and target:
        # source: first self.max_length tokens
        # target: last self.max_length tokens 
        source_sequence = full_seq[:-1]
        target_sequence = full_seq[1:]
        
        # Create key padding mask for the source.
        # Here, True indicates a pad token and thus a masked position.
        key_padding_mask = [tok == self.pad_id for tok in source_sequence]
        
        # Convert to tensors
        source_sequence = torch.tensor(source_sequence, dtype=torch.long)
        target_sequence = torch.tensor(target_sequence, dtype=torch.long)
        key_padding_mask = torch.tensor(key_padding_mask, dtype=torch.bool)
        
        return {
            "source_sequence": source_sequence, 
            "target_sequence": target_sequence,
            "key_padding_mask": key_padding_mask,
        }


if __name__ == "__main__":
    from config import config
    from tokenizers import Tokenizer
    from datasets import load_dataset

    # Sanity check the dataset class
    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    idx = 1
    config.max_len = 64  # For testing purposes
    dataset = QADataset(config, tokenizer)

    data_item = dataset[idx]
    source = data_item["source_sequence"]
    target = data_item["target_sequence"]
    key_padding_mask = data_item["key_padding_mask"]

    print("Source sequence shape:", source.shape)
    print("Target sequence shape:", target.shape)
    print("Key padding mask shape:", key_padding_mask.shape)

    print("Source sequence:", source)
    print("Target sequence:", target)
    print("Key padding mask:", key_padding_mask)

    # When decoding the target, filter out pad tokens using config.pad_token
    decoded_source = tokenizer.decode(source.tolist(), skip_special_tokens=False)
    decoded_target = tokenizer.decode(
        [tok for tok in target.tolist() if tok != dataset.pad_id],
        skip_special_tokens=False
    )
    print("Decoded source sequence:", decoded_source)
    print("Decoded target sequence:", decoded_target)
