# !pip install datasets
# !pip install torch torchvision torchaudio
# !pip install numpy
# !pip install pandas
# !pip install matplotlib
# !pip install accelerate
# !pip install transformers
# !pip install -U scikit-learn scipy matplotlib
# !pip install nltk
# !pip install --upgrade nltk



import gc
import torch
import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import nltk
import os
import random

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, MBartConfig
from nltk.translate import bleu_score
from nltk.tokenize import sent_tokenize


from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer

nltk.download('wordnet')
nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)


def print_gpu_memory_usage(step):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        free = torch.cuda.get_device_properties(0).total_memory / (1024**3) - reserved
        print(f"{step}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Free: {free:.2f} GB")
    else:
        print(f"{step}: CUDA not available")


def read_file(file_path, max_lines=10010):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()[:max_lines]]

print_gpu_memory_usage("Before loading data")

# My normal dataset

# text = "/kaggle/input/for-test-work/rus.txt"
# with open(text) as file:
#     lines = file.read().split("\n")[:-1]
# pairs = []
# for line in lines:
#     english, russian = line.split("\t")[:2]
#     russian = "[start] " + russian + " [end]"
#     pairs.append((english, russian))

# max_lines = 100
# en_lines = df['en'].tolist()[:max_lines]

# My dataset for data shift

# dataset_dir = '/kaggle/input/shifts-dataset/'
# en_file = os.path.join(dataset_dir, 'UNv1.0.en-ru.en')
# ru_file = os.path.join(dataset_dir, 'UNv1.0.en-ru.ru')
# ids_file = os.path.join(dataset_dir, 'UNv1.0.en-ru.ids')

# print(f"English file: {len(en_file)}")
# print(f"Russian file: {len(ru_file)}")
# print(f"IDs file: {len(ids_file)}")

# en_lines = read_file(en_file)
# ru_lines = read_file(ru_file)
# ids_lines = read_file(ids_file)

# print(f"English lines: {len(en_lines)}")
# print(f"Russian lines: {len(ru_lines)}")
# print(f"IDs lines: {len(ids_lines)}")


# en_dict = {f"{i}:{j}": line for i, line in enumerate(en_lines, start=1) for j in range(1, 3)}
# ru_dict = {f"{i}:{j}": line for i, line in enumerate(ru_lines, start=1) for j in range(1, 3)}


# data = []
# for line in ids_lines:
#     parts = line.split()


#     en_indices = [p.split(":")[1:] for p in parts if p.startswith("en:")]
#     ru_indices = [p.split(":")[1:] for p in parts if p.startswith("ru:")]


#     en_indices = [f"{i}:{j}" for i, j in en_indices]
#     ru_indices = [f"{i}:{j}" for i, j in ru_indices]


#     en_texts = [en_dict[idx] for idx in en_indices if idx in en_dict]
#     ru_texts = [ru_dict[idx] for idx in ru_indices if idx in ru_dict]


#     if en_texts and ru_texts:
#         data.append({
#             'en': " ".join(en_texts),
#             'ru': " ".join(ru_texts) 
#         })

# print(f"Total dataset size: {len(data)}")

# df = pd.DataFrame(data)



dataset_dir = '/kaggle/input/shifts-dataset/'
en_file = os.path.join(dataset_dir, 'UNv1.0.en-ru.en')
ru_file = os.path.join(dataset_dir, 'UNv1.0.en-ru.ru')


en_lines = read_file(en_file)
ru_lines = read_file(ru_file)

assert len(en_lines) == len(ru_lines), "EN and RU line counts do not match"

print(f"Original line pairs: {len(en_lines)}")


augmented_data = []

for en_text, ru_text in zip(en_lines, ru_lines):
    en_sents = sent_tokenize(en_text.strip())
    ru_sents = sent_tokenize(ru_text.strip())


    for en_sent, ru_sent in zip(en_sents, ru_sents):
        if en_sent and ru_sent:
            augmented_data.append({'en': en_sent, 'ru': ru_sent})

print(f"Total augmented examples: {len(augmented_data)}")


df = pd.DataFrame(augmented_data)



for i in range(90, 100):
    if i < len(df):
        print(f"\nRow {i+1}:")
        print(f"English: {df.iloc[i]['en']}")
        print(f"Russian: {df.iloc[i]['ru']}")


#print(df.head())
#print(df.info())

# print(df['length_diff'].describe())
# print(df['is_exact_match'].value_counts())

# df = pd.DataFrame(pairs, columns=['en', 'ru']) # Normal dataset


#check_dataset(data, num_samples=10)

print_gpu_memory_usage("After loading data")



dataset = datasets.Dataset.from_pandas(df)


# train_size = int(len(dataset) * 0.8)
# eval_size = len(dataset) - train_size

# dataset_train = dataset.select(range(train_size))
# dataset_eval = dataset.select(range(train_size, len(dataset))

print_gpu_memory_usage("After loading data")

dataset = datasets.Dataset.from_pandas(df)

dataset_train = dataset.select(range(0, 10000))
dataset_eval = dataset.select(range(0, 10000))

print(f"Train dataset size: {len(dataset_train)}")
print(f"Eval dataset size: {len(dataset_eval)}")


tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50', src_lang="en_XX", tgt_lang="ru_RU", return_tensors="pt")
def preprocess_function(examples):
     inputs = examples['en']
     targets = examples['ru']
     model_inputs = tokenizer(examples['en'], 
                            text_target=examples['ru'], padding=True, truncation=True)
     return model_inputs
    
def collate_fn(batch):
     inputs = [item['en'] for item in batch]
     targets = [item['ru'] for item in batch]
     model_inputs = tokenizer(inputs, max_length=256, padding=True,
                             truncation=True, return_tensors="pt")
     with tokenizer.as_target_tokenizer():
         labels = tokenizer(targets, max_length=256, padding=True,
                            truncation=True, return_tensors="pt")
     model_inputs["labels"] = labels["input_ids"]
     return model_inputs

model_inputs_train = dataset_train.map(preprocess_function,
                                        batched=True)
model_inputs_test = dataset_eval.map(preprocess_function,
                                        batched=True)
train_loader = DataLoader(dataset=model_inputs_train,
                            collate_fn=collate_fn, batch_size=10, pin_memory=True, drop_last=False)
validation_loader = DataLoader(dataset=model_inputs_test,
                                collate_fn=collate_fn, batch_size=10, pin_memory=True, drop_last=False)



def create_model_with_dropout(dropout_rate):
     config = MBartConfig.from_pretrained('facebook/mbart-large-50-many-to-many-mmt',
                                            dropout=dropout_rate, attention_dropout=dropout_rate)
     model_dir = "/kaggle/input/mbart-large-50-70000to170000-80000to100000-5epochs/pytorch/default/1/results/to/save/model_dropout_result"
     model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt', config=config)
     # model = MBartForConditionalGeneration.from_pretrained("path/to/save/model_dropout_0.1", config=config)
     return model
    
def train(model, iterator, optimizer, criterion, clip):
     model.train()
     epoch_loss = 0.0
     for i, batch in enumerate(iterator):
         # print(f"Batch type: {type(batch)}")
         # if not isinstance(batch, dict):
         # print(f"Unexpected batch type: {type(batch)}")
         # continue

         inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
         labels = batch['labels'].to(device)
         output = model(**inputs, labels=labels)
         loss = output.loss
         optimizer.zero_grad()
         loss.backward()
         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
         optimizer.step()
         epoch_loss += loss.item()
         torch.cuda.empty_cache()
         return epoch_loss / len(iterator)
         
def evaluate_model(model, iterator, criterion):
     model.eval()
     epoch_loss = 0.0
     predictions = []
     references = []
     with torch.no_grad():
         for batch in iterator:
            inputs = {key: value.to(device) for key, value in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)

            output = model(**inputs, labels=labels)
            logits = output.logits
            
            generated_outputs = model.generate(
                **inputs,
                max_length=200,
                num_beams=5,
                do_sample=True,
                temperature=0.3,
                forced_bos_token_id=tokenizer.lang_code_to_id["ru_RU"]
            )

            batch_predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in generated_outputs]
            batch_references = [tokenizer.decode(ref, skip_special_tokens=True) for ref in labels]

            predictions.extend(batch_predictions)
            references.extend(batch_references)

            input_texts = [tokenizer.decode(inp, skip_special_tokens=True) for inp in inputs['input_ids']]
            # for inp, pred, ref in zip(input_texts, batch_predictions, batch_references):
            #     print(f"Входное предложение: {inp}")
            #     print(f"Сгенерированный перевод: {pred}")
            #     print(f"Референсный перевод: {ref}\n")

            # Выбираем 5 случайных примеров для вывода
            indices = random.sample(range(len(input_texts)), min(3, len(input_texts)))
    
            # for i in indices:
            #     print(f"Входное предложение: {input_texts[i]}")
            #     print(f"Сгенерированный перевод: {batch_predictions[i]}")
            #     print(f"Референсный перевод: {batch_references[i]}\n")

            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            loss = criterion(logits, labels)
            epoch_loss += loss.item()

            del inputs, labels, output, logits, generated_outputs, loss
            torch.cuda.empty_cache()

            assert len(references) == len(predictions)
            predictions_tokenized = [tokenizer.tokenize(pred) for pred in predictions]
            references_tokenized = [tokenizer.tokenize(ref) for ref in references]

     # BLEU score
     predictions_flat = [token for tokens in predictions_tokenized for token in tokens]
     references_flat = [token for tokens in references_tokenized for token in tokens]
     smoothing_function = SmoothingFunction().method4
     # bleu = bleu_score.corpus_bleu([[references_flat]], [predictions_flat], smoothing_function=smoothing_function)
     bleu = bleu_score.corpus_bleu([[references_flat]], [predictions_flat], smoothing_function=smoothing_function)
 
    # METEOR
     # meteor_avg_score = meteor_score([references_flat], predictions_flat)
     min_length = min(len(predictions_flat), len(references_flat))
     predictions_flat = predictions_flat[:min_length]
     references_flat = references_flat[:min_length]

     # F1-score
     f1 = f1_score(references_flat, predictions_flat, average='weighted')

     # Precision
     precision = precision_score(references_flat, predictions_flat, average='micro')

     # Recall
     recall = recall_score(references_flat, predictions_flat, average='micro')

     # ROUGE score
 # rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
 # rouge_scores = {key: [] for key in rouge_scorer_obj.score('', '').keys()}
 # for ref, pred in zip(references_tokenized, predictions_tokenized):
 # ref_str = ' '.join(ref)
 # pred_str = ' '.join(pred)
 # scores = rouge_scorer_obj.score(ref_str, pred_str)
 # for key in rouge_scores:
 # rouge_scores[key].append(scores[key].fmeasure)
 # avg_rouge_scores = {key: sum(scores) / len(scores) for key, scores in rouge_scores.items()}

     return epoch_loss / len(iterator), bleu, f1, precision, recall 

init_token = '<sos>'
eos_token = '<eos>'

def epoch_time(start_time, end_time):
     elapsed_time = end_time - start_time
     elapsed_mins = int(elapsed_time / 60)
     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
     return elapsed_mins, elapsed_secs
    
def train_and_evaluate(dropout_rate):
    model = create_model_with_dropout(dropout_rate)
    model.to(device)
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1) # May be -2
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    N_EPOCHS = 100
    CLIP = 1
    best_valid_loss = float('inf')
    writer = SummaryWriter()

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = 0.0
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss, bleu_score_value, f1_score_value, precision, recall = evaluate_model(model, validation_loader, criterion)
    
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        #if valid_loss < best_valid_loss:
        # best_valid_loss = valid_loss

        model.save_pretrained(f"/kaggle/working/to/save/model_dropout_result")
        tokenizer.save_pretrained(f"/kaggle/working/to/save/tokenizer/checkpoint_mbart_50k")

        writer.add_scalar("Train Loss", train_loss, epoch + 1)
        writer.add_scalar("Train PPL", math.exp(train_loss), epoch + 1)
        writer.add_scalar("Val. Loss", valid_loss, epoch + 1)
        writer.add_scalar("Val. PPL", math.exp(valid_loss), epoch + 1)
        writer.add_scalar("BLEU Score", bleu_score_value, epoch + 1)
        writer.add_scalar("F1 Score", f1_score_value, epoch + 1)
        writer.add_scalar("Precision", precision, epoch + 1)
        writer.add_scalar("Recall", recall, epoch + 1)
        # for metric_name, avg_score in avg_rouge_scores.items():
        # writer.add_scalar(f"Avg_ROUGE/{metric_name}", avg_score,epoch + 1)
        # writer.add_scalar("Meteor_avg_score", meteor_avg_score,epoch + 1)
        print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')
        print(f'\t BLEU Score: {bleu_score_value:.3f}')
        print(f'\t F1 Score: {f1_score_value:.3f}')
        print(f'\t Precision: {precision:.3f}')
        print(f'\t Recall: {recall:.3f}')
        # print(f'\t Avg_rouge_scores:', avg_rouge_scores)
        # print(f'\t Meteor_avg_score: {meteor_avg_score:.3f}')
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_np = param.detach().cpu().numpy()
                writer.add_histogram(name, param_np, global_step=epoch)
    
    words = list(tokenizer.get_vocab().keys())
    word_embedding = model.model.shared.weight
    writer.add_embedding(word_embedding, metadata=words, tag='word embedding')

    writer.close()

dropout_rates = [0.3]

for rate in dropout_rates:
     print(f"Training with dropout rate: {rate}")
     train_and_evaluate(rate)



     