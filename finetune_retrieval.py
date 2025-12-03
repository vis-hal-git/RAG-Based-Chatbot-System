# finetune_retrieval.py
"""
Simple stub / starting point for retrieval fine-tuning.
This script demonstrates how to create (question, positive_chunk, negative_chunk)
training examples and then fine-tune a sentence-transformers model offline.
You will need to adapt for your compute environment.
"""
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

def prepare_pairs(chunks, qas):
    # qas: list of (question, positive_chunk_index)
    examples = []
    for q, pos_idx in qas:
        pos = chunks[pos_idx]["content"]
        # sample random negative
        neg_idx = (pos_idx+1) % len(chunks)
        neg = chunks[neg_idx]["content"]
        examples.append(InputExample(texts=[q, pos, neg]))
    return examples

def train_model(examples, model_name="all-MiniLM-L6-v2", epochs=2, batch_size=8):
    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=100)
    model.save("finetuned_retriever")
