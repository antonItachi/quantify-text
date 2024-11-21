import torch
from torch.utils.data import TensorDataset, DataLoader
import regex as re


def preprocess_text(dataset, min_length=20):
    """
    Basic preprocess of the text.
    """
    def clean_text(text):
        text = re.sub(r"<.*?>", "", text)  # Удаление HTML
        text = re.sub(r"\s+", " ", text)  # Удаление лишних пробелов
        text = re.sub(r"[^\w\s.,!?;'-]", "", text)  # Удаление спецсимволов
        return text.strip()

    cleaned_data = [clean_text(example['text']) for example in dataset]

    # Length filtration
    filtered_data = [text for text in cleaned_data if len(text.split()) >= min_length]

    return filtered_data

def split_in_sequence(filtered_data, tokenizer, batch_size=4, context_length=1024, overlap=0):
    current_sequence = []
    input_batch = []

    for element in filtered_data:
        outputs = tokenizer(
            element,
            return_tensors="pt",
        )
        input_ids = outputs["input_ids"].squeeze(0).tolist()

        current_sequence.extend(input_ids)

        while len(current_sequence) >= context_length:
            input_batch.append(torch.tensor(current_sequence[:context_length]))
            current_sequence = current_sequence[context_length - overlap:]

    if len(current_sequence) > 0:
        padding_length = context_length - len(current_sequence)
        padded_sequence = torch.tensor(
            current_sequence + [tokenizer.eos_token_id] * padding_length
        )
        input_batch.append(padded_sequence)

    if not input_batch:
        raise ValueError("No valid sequences found. Check your filtered_data or context_length.")

    input_data = torch.stack(input_batch)
    labels = input_data.clone()

    dataset = TensorDataset(input_data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
