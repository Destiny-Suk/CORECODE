import torch
import json, os, logging, pickle, argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='Initial learning rate') 
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=12, metavar='E', help='number of epochs')
    parser.add_argument('--model', default='rob', help='which model rob| robsq | span')
    parser.add_argument('--cuda', type=int, default=0, metavar='C', help='cuda device')
    # parser.add_argument('--fold', type=int, default=0, metavar='F', help='which fold')
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--validation_file", type=str)
    parser.add_argument("--test_file", type=str)
 
    args = parser.parse_args()
    print(args)
    return args

args = parse_config()
checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

max_length = 384
stride = 128

train_data = [json.loads(line) for line in open(args.train_file).readlines()]
dev_data = [json.loads(line) for line in open(args.validation_file).readlines()]
test_data = [json.loads(line) for line in open(args.test_file).readlines()]

def train_collote_fn(batch_samples):
    batch_question, batch_context, batch_answers = [], [], []
    for sample in batch_samples:
        batch_context.append(sample['context'])
        batch_question.append(sample['question'])
        batch_answers.append(sample['answer'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
        return_tensors="pt"
    )
    
    offset_mapping = batch_data.pop('offset_mapping')
    sample_mapping = batch_data.pop('overflow_to_sample_mapping')

    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answer = batch_answers[sample_idx]
        start_char = answer['answer_start'][0]
        end_char = answer['answer_start'][0] + len(answer['text'][0])
        sequence_ids = batch_data.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    return batch_data, torch.tensor(start_positions), torch.tensor(end_positions)
 
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=train_collote_fn)

def test_collote_fn(batch_samples):
    batch_id, batch_question, batch_context = [], [], []
    for sample in batch_samples:
        batch_id.append(sample['id'])
        batch_question.append(sample['question'])
        batch_context.append(sample['context'])
    batch_data = tokenizer(
        batch_question,
        batch_context,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length", 
        return_tensors="pt"
    )
    
    offset_mapping = batch_data.pop('offset_mapping').numpy().tolist()
    sample_mapping = batch_data.pop('overflow_to_sample_mapping')
    example_ids = []

    for i in range(len(batch_data['input_ids'])):
        sample_idx = sample_mapping[i]
        example_ids.append(batch_id[sample_idx])

        sequence_ids = batch_data.sequence_ids(i)
        offset = offset_mapping[i]
        offset_mapping[i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]
    return batch_data, offset_mapping, example_ids

valid_dataloader = DataLoader(dev_data, batch_size=8, shuffle=False, collate_fn=test_collote_fn)