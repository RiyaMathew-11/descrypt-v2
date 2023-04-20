import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from models import SiameseBERT
from csv import writer
import pandas as pd

DATA_PATH = './data/data-draft-v2.csv'
OUT_PATH = './results/bert-base-multilingual-cased-2.csv'

qna = pd.read_csv(DATA_PATH)

model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)


# Initialize the Siamese BERT model from models
model = SiameseBERT(model_name)

for num in range(len(qna)):
    # Example text pairs
    text1 = qna['Model_Answer'][num]
    text2 = qna['User_Answer'][num]

    inputs1 = tokenizer(text1, return_tensors="pt", padding=True,
                        truncation=True, max_length=128)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True,
                        truncation=True, max_length=128)

    # Compute the similarity score
    with torch.no_grad():
        similarity_score = model(
            input_ids1=inputs1["input_ids"],
            attention_mask1=inputs1["attention_mask"],
            input_ids2=inputs2["input_ids"],
            attention_mask2=inputs2["attention_mask"]
        )
        print("\n\nSimilarity score:", round(similarity_score.item(), 2))

        entry = [text1, text2, round(similarity_score.item(), 2)]

        with open(OUT_PATH, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(entry)
            write_obj.close()
