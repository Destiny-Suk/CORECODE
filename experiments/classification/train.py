import logging
import torch
import argparse
import numpy as np

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, Union

import datasets
import numpy as np
import torch
from datasets import load_dataset
from data_preprocessing import Task1_Dataloader, get_dataLoader
import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from sklearn.metrics import accuracy_score
#from dataclass import load_dataset
from models import SeperateTaskForClassification
from datasets import load_dataset, load_metric
from datacollator import CustomCollatorWithPadding
from transformers import DataCollatorWithPadding, DataCollatorForTokenClassification
from transformers import  AutoTokenizer, AutoConfig, set_seed
from transformers import TrainingArguments, Trainer
logger = logging.getLogger(__name__) # https://github.com/tjunlp-lab/TGEA/blob/main/Diagnosis_tasks/train.py
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--number_of_gpu", type=int, help="Number of available GPUs.")

    parser.add_argument("--batch_size_per_gpu", type=int, default=32,help='batch size for each gpu.')
    parser.add_argument("--test_batch_size_per_gpu", type=int, default=32, help='test batch size for each gpu.')
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation step.")
    parser.add_argument("--effective_batch_size", type=int,
                        help='effective_bsz = batch_size_per_gpu x number_of_gpu x gradient_accumulation_steps')
    parser.add_argument("--total_steps", type=int,
                        help='total effective training steps for pre-training stage')
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--num_labels", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--do_train", type=bool, default=False)
    parser.add_argument("--do_eval", type=bool, default=False)
    parser.add_argument("--do_test", type=bool, default=False)
    parser.add_argument("--use_focal", type=bool, default=False)
    parser.add_argument("--focal_alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42, help='training seed')


    return parser.parse_args()

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


if __name__ == '__main__':
    cuda_available =  torch.cuda.is_available()
    if cuda_available:
        print("Cuda is available.")
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print("Using Multi-GPU training, number of GPU is {}".format(torch.cuda.device_count()))
        else:
            print("Using single GPU training.")
    else:
        pass
    args = parse_config()
    device = torch.device('cuda')
    set_seed(args.seed)

    batch_size_per_gpu, gradient_accumulation_steps, number_of_gpu = \
    args.batch_size_per_gpu, args.gradient_accumulation_steps, args.number_of_gpu
    # effective_batch_size = batch_size_per_gpu * gradient_accumulation_steps * number_of_gpu
    # number_of_gpu:

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_files = {'train': args.train_path, 'validation': args.dev_path, 'test': args.test_path}
    dataset = load_dataset('json', data_files=data_files, cache_dir='./')
    # train_dataset = Task1_Dataloader(args.train_path)
    # dev_dataset = Task1_Dataloader(args.dev_path)
    # test_dataset = Task1_Dataloader(args.test_path)

    pre_metrics = None
    # metric_precision = load_metric("precision")
    # metric_recall = load_metric("recall")
    # metric_f1 = load_metric("f1")
    # metric_accuracy = load_metric("accuracy")
    metric_precision = load_metric("mymetrics/precision")
    metric_recall = load_metric("mymetrics/recall")
    metric_f1 = load_metric("mymetrics/f1")
    metric_accuracy = load_metric("mymetrics/accuracy")
    
    # =================== task1 ==================
    if args.task_name == 'task1':
        input_type = 'pooled_cls'
        classification_type = 'sequence'
        num_labels = 3
        # Preprocessing the datasets.
        ending_names = [f"choice{i}" for i in range(3)] #['choice0', 'choice1', 'choice2']
        context_name = "context"
        def preprocess_function(examples):
            first_sentences = [[context] * 3 for context in examples[context_name]] # len(dataset)个，每个是对话上下文 重复3遍，三个list，list[list]
            second_sentences = [
                [f"{examples[end][i]}" for end in ending_names] for i in range(len(examples[context_name]))
            ]  # len(dataset)个，每个是3个选项，list[list]

            # Flatten out
            first_sentences = sum(first_sentences, []) # 3*len(dataset)个
            second_sentences = sum(second_sentences, []) # 3*len(dataset)个
            # 即变成3*len(dataset)个，分别是：
            # 对话1 对话1的选项1
            # 对话1 对话1的选项2
            # 对话1 对话1的选项3
            # 对话2 对话2的选项1
            # ... ...

            # Tokenize
            # Use the choices as the first item in tokenizer to avoid choice trimming if tokenized length exceeds 512
            max_seq_length = 300
            tokenized_examples = tokenizer(
                second_sentences,
                first_sentences,
                truncation=True,
                stride=128,
                max_length=384,
                # padding="max_length" if data_args.pad_to_max_length else False,
                # padding=False,
                padding='max_length',
                return_tensors="pt",
            )
            # Un-flatten
            return {k: [v[i : i + 3] for i in range(0, len(v), 3)] for k, v in tokenized_examples.items()}
        
        tokenized_datasets = dataset.map(preprocess_function, batched=True)
        for ids in tokenized_datasets["train"]["input_ids"][0]:
            print(tokenizer.decode(ids))
        # tokenized_datasets['train'][2]:
        # 'ID':
        # 'naturalconv_4831_0'
        # 'context':
        # '对话中的<MASK>处应填入什么？ \\n 对话: A: 嗨你好啊。 <utt> B: 嗯呢，你好，你也来看教职工运动会嘛？ <utt> A: 嗯呢，无聊了，想看球玩玩，就过来了。 <utt> B: 怎么没看国足打叙利亚嘛？就这之后里皮宣布辞职的那场。 <utt> A: 诶，不想看了，里皮二进宫还是<MASK>，只好自己辞职呗。 <utt> B: 其实里皮二进宫也有一点成绩的，打了七场比赛拿了5场胜利呢。还算行吧。 <utt> A: 但是我觉的，里皮都不谈国足出线形势。直接辞职，这还不够明了嘛？ <utt> B: 诶，这么多年了，要他有何用啊，1.5个亿的年薪啊，真的是打水漂了。 <utt> A: 有点难受啊，国足不行延续到了现在，估计有生之年看不到他进世界杯了。 <utt> B: 哈哈哈，对啊，都带不动了。 <utt> A: 哎，有点难受，你是不是也是因为昨天看伤心了，今天出来娱乐一下。 <utt> B: 差不多吧，在呆在宿舍我能郁闷死。 <utt> A: 是啊，我也是，要不我们去借个篮球，打一场转移一下注意力。 <utt> B: 哈哈哈，可以啊，你住哪栋楼，要一起回去换个衣服吗？ <utt> A: 一号楼，你呢。 <utt> B: 我三号楼，我去借个球，你先回去换衣服。 <utt> A: 行，那我先走一步。 <utt> B: 诶，先别急，加个微信，不然找不到人，哈哈哈。 <utt> A: 哦哦哦，你不说我都忘记了。来。 <utt> B: 嗯嗯。'
        # 'choice0':
        # '带的动国足'
        # 'choice1':
        # '带不起来国足'
        # 'choice2':
        # '带不起女排'
        # 'label': 1
        # 'input_ids': [[101, 2372, 4638, 1220, 1744, 6639, 102, 2190, 6413, ...], [101, 2372, 679, 6629, 3341, 1744, 6639, 102, 2190, ...], [101, 2372, 679, 6629, 1957, 2961, 102, 2190, 6413, ...]]
        # 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 1, 1, ...], [0, 0, 0, 0, 0, 0, 0, 0, 1, ...], [0, 0, 0, 0, 0, 0, 0, 1, 1, ...]]
        # 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, ...], [1, 1, 1, 1, 1, 1, 1, 1, 1, ...], [1, 1, 1, 1, 1, 1, 1, 1, 1, ...]]
        #         tokenized_datasets = tokenized_datasets.remove_columns(books_dataset["train"].column_names)


        # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

        def compute_metrics(eval_pred):
            results = {}
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            f1 = metric_f1.compute(predictions=predictions, references=labels)
            accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
            precision = metric_precision.compute(predictions=predictions, references=labels)
            recall = metric_recall.compute(predictions=predictions, references=labels)
            results.update(accuracy)
            results.update(precision)
            results.update(recall)
            results.update(f1)
            return results
        
        # model = SeperateTaskForClassification(model_name,
        #              input_type,
        #              classification_type,
        #              num_labels,
        #              use_focal=args.use_focal,
        #              focal_alpha=args.focal_alpha)
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name,

    )
        metrics = compute_metrics

    # =================== b1 ==================
    if args.task_name == 'ErroneousDetection':
        input_type = 'pooled_cls'
        classification_type = 'sequence'
        num_labels = 2

        def process_label(example):
            example['label'] = 0 if example['label'] == '正确' else 1
            return example
        dataset = dataset.map(process_label)
        # dataset['train'][0]:
        # 'text':'在完成游戏时,会有各种不同的小游戏和一些文字来提示玩家如何操作。'
        # 'label':0

        def tokenizer_function(example):
            return tokenizer(example['text'])
        tokenized_datasets = dataset.map(tokenizer_function, batched=True)
        # tokenized_datasets['train'][0]:
        # 'text':'在完成游戏时,会有各种不同的小游戏和一些文字来提示玩家如何操作。'
        # 'label':0
        # 'input_ids':[101, 1762, 2130, 2768, 3952, 2767, 3198, 117, 833, 3300, 1392, 4905, 679, 1398, ...]
        # 'token_type_ids':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]
        # 'attention_mask':[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        def compute_metrics(eval_pred):
            results = {}
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            f1 = metric_f1.compute(predictions=predictions, references=labels)
            accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
            precision = metric_precision.compute(predictions=predictions, references=labels)
            recall = metric_recall.compute(predictions=predictions, references=labels)
            results.update(accuracy)
            results.update(precision)
            results.update(recall)
            results.update(f1)
            return results
        
        model = SeperateTaskForClassification(model_name,
                     input_type,
                     classification_type,
                     num_labels,
                     use_focal=args.use_focal,
                     focal_alpha=args.focal_alpha)
        metrics = compute_metrics

    # =================== b2 ==================
    elif args.task_name == 'MiSEWDetection':

        input_type = None
        classification_type = 'token'
        num_labels = 2

        def process_function(example):
            tokenized_text = tokenizer(example['text'].strip().lstrip().split(' '), is_split_into_words=True, return_offsets_mapping=True)
            raw_label = example.pop('label')
            label_ids = []
            word_ids = tokenized_text.word_ids()
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(raw_label[word_idx])
            tokenized_text['labels'] = label_ids
            return tokenized_text
        tokenized_datasets = dataset.map(process_function)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        def compute_metrics_mapping(trainer, dataset):
            predictions, labels, _ = trainer.predict(dataset)
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            offsets = dataset['offset_mapping']
            total_f1 = []
            total_accuracy = []
            total_precision = []
            total_recall = []
            text = dataset['text']
            nums = 0
            for i in range(len(offsets)):

                true_prediction = [p for (p,l) in zip(predictions[i], labels[i]) if l != -100]
                true_label= [l for (p,l) in zip(predictions[i], labels[i]) if l != -100]
                offset = [o for o in offsets[i] if (o[1] - o[0]) != 0]
                mapping_prediction = []
                mapping_label = []
                for j in range(len(true_label)):
                    mapping_prediction.extend([true_prediction[j]] * (offset[j][1] - offset[j][0]))
                    mapping_label.extend([true_label[j]] * (offset[j][1] - offset[j][0]))

                if  len(''.join(text[i].split(' '))) != len(mapping_prediction):
                    nums += 1
                f1 = metric_f1.compute(predictions=mapping_prediction, references=mapping_label)
                accuracy = metric_accuracy.compute(predictions=mapping_prediction, references=mapping_label)
                precision = metric_precision.compute(predictions=mapping_prediction, references=mapping_label)
                recall = metric_recall.compute(predictions=mapping_prediction, references=mapping_label)

                total_f1.append(f1['f1'])
                total_accuracy.append(accuracy['accuracy'])
                total_precision.append(precision['precision'])
                total_recall.append(recall['recall'])
            result = {
                      'accuracy': np.mean(total_accuracy),
                      'precision': np.mean(total_precision),
                      'recall':np.mean(total_recall),
                      'f1': np.mean(total_f1),
                      }
            return result

        model = SeperateTaskForClassification(model_name,
                     input_type,
                     classification_type,
                     num_labels,
                     use_focal=args.use_focal,
                     focal_alpha=args.focal_alpha)
        metrics = None

    # =================== b3 ==================
    elif 'spandetection' in args.task_name:
        input_type = None
        classification_type = 'token'
        num_labels = 3

        def process_function(example):
            tokenized_text = tokenizer(example['text'].strip().lstrip().split(' '), is_split_into_words=True,
                                       return_offsets_mapping=True)
            raw_label = example.pop('label')
            label_ids = []
            misew_ids = []
            word_ids = tokenized_text.word_ids()
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                    misew_ids.append(-100)
                else:
                    # 2 errorneous span
                    # 1 misew span
                    # 0 other span
                    if raw_label[word_idx] == 2:
                        label_ids.append(1)
                        misew_ids.append(1)
                    elif raw_label[word_idx] == 1:
                        label_ids.append(0)
                        misew_ids.append(0)
                    else:
                        label_ids.append(-100)
                        misew_ids.append(-100)
            tokenized_text['labels'] = label_ids
            tokenized_text['misew_labels'] = misew_ids
            return tokenized_text

        tokenized_datasets = dataset.map(process_function)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        def compute_metrics_mapping(trainer, dataset):
            predictions, labels, _ = trainer.predict(dataset)
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            offsets = dataset['offset_mapping']
            misew_labels = dataset['misew_labels']
            total_f1 = []
            total_accuracy = []
            total_precision = []
            total_recall = []
            total_f1_misew = []
            total_accuracy_misew = []
            total_precision_misew = []
            total_recall_misew = []
            text = dataset['text']
            for i in range(len(offsets)):
                true_prediction = [p for (p, ml) in zip(predictions[i], misew_labels[i]) if ml != -100]
                true_label = [l for (l, ml) in zip(labels[i], misew_labels[i]) if ml != -100]

                offset = [o for o in offsets[i] if (o[1] - o[0]) != 0]
                mapping_prediction = []
                mapping_label = []
                for j in range(len(true_label)):
                    mapping_prediction.extend([true_prediction[j]] * (offset[j][1] - offset[j][0]))
                    mapping_label.extend([true_label[j]] * (offset[j][1] - offset[j][0]))
                f1 = metric_f1.compute(predictions=mapping_prediction, references=mapping_label)
                accuracy = metric_accuracy.compute(predictions=mapping_prediction, references=mapping_label)
                precision = metric_precision.compute(predictions=mapping_prediction, references=mapping_label)
                recall = metric_recall.compute(predictions=mapping_prediction, references=mapping_label)
                '''
                true_prediction_misew = [p for (p, ml) in zip(predictions[i], misew_labels[i]) if ml != -100]
                true_label_misew = [l for (l, ml) in zip(labels[i], misew_labels[i]) if ml != -100]

                for idx in range(len(true_prediction_misew)):
                    if true_prediction_misew[idx] == 2:
                        true_prediction_misew[idx] = 0
                    else:
                        true_prediction_misew[idx] = 1
                    if true_label_misew[idx] == 2:
                        true_label_misew[idx] = 0
                    else:
                        true_label_misew[idx] = 1

                #offset = [o for o in offsets[i] if (o[1] - o[0]) != 0]
                mapping_prediction_misew = []
                mapping_label_misew = []
                for j in range(len(true_label_misew)):
                    mapping_prediction_misew.extend([true_prediction_misew[j]] * (offset[j][1] - offset[j][0]))
                    mapping_label_misew.extend([true_label_misew[j]] * (offset[j][1] - offset[j][0]))
                f1_misew = metric_f1.compute(predictions=mapping_prediction_misew, references=mapping_label_misew)
                accuracy_misew = metric_accuracy.compute(predictions=mapping_prediction_misew, references=mapping_label_misew)
                precision_misew = metric_precision.compute(predictions=mapping_prediction_misew, references=mapping_label_misew)
                recall_misew = metric_recall.compute(predictions=mapping_prediction_misew, references=mapping_label_misew)
                '''

                total_f1.append(f1['f1'])
                total_accuracy.append(accuracy['accuracy'])
                total_precision.append(precision['precision'])
                total_recall.append(recall['recall'])
                '''
                total_f1_misew.append(f1_misew['f1'])
                total_accuracy_misew.append(accuracy_misew['accuracy'])
                total_precision_misew.append(precision_misew['precision'])
                total_recall_misew.append(recall_misew['recall'])
                '''
            result = {
                      'accuracy': np.mean(total_accuracy),
                      'precision': np.mean(total_precision),
                      'recall':np.mean(total_recall),
                      'f1': np.mean(total_f1),
                        #
                        # 'misew_accuracy': np.mean(total_accuracy_misew),
                        # 'misew_precision': np.mean(total_precision_misew),
                        # 'misew_recall': np.mean(total_recall_misew),
                        # 'misew_f1': np.mean(total_f1_misew),

                      }
            return result

        model = SeperateTaskForClassification(model_name,
                                              input_type,
                                              classification_type,
                                              num_labels,
                                              use_focal=args.use_focal,
                                              focal_alpha=args.focal_alpha)
        metrics = None # compute_metrics

    # =================== b4 ==================
    # args.task_name: 'ErroneousClassification_l1'
    elif 'ErroneousClassification' in args.task_name:
        #input_type = 'span'
        input_type = 'pooled_cls'
        classification_type = 'sequence'
        if args.task_name[-1] == '1':
            label_maps = {'搭配错误':0,
                          '残缺错误':1,
                          '成分多余':2,
                          '语篇错误':3,
                          '常识错误':4}

        num_labels = len(label_maps)

        def process_function(example):
            # tokenized_text: [错误span前面的文字，错误span，错误span后面的文字]
            # is_split_into_words默认为False，需要经过tokenize。而这里设置is_split_into_words=True表示已经pre-tokenized了，即认为把 错误span前面的文字-错误span-错误span后面的文字 分开就是 pre-tokenized
            tokenized_text = tokenizer(example['text'].strip().lstrip().split(' '), is_split_into_words=True, return_offsets_mapping=True)
            # tokenized_text.tokens(): ['[CLS]', '而', '且', '这', '个', '时', '间', '段', '正', '是', '我', '们', '比', '较', ...]
            
            span_id = example['span'] # 表示错误span前面有没有文字，有的话为1；没有的话为0
            error_span_mask = []
            for i in tokenized_text.word_ids(): # [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, ...]
                if i == span_id:
                    error_span_mask.append(1)
                else:
                    error_span_mask.append(0)  # error_span_mask:error_span_mask错误span的mask，为1为错误的span
            # error_span_mask: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, ...]
            tokenized_text['error_span_mask'] = error_span_mask
            if args.task_name[-1] == '1':
                tokenized_text['labels'] = label_maps[example['class1']]
            elif args.task_name[-1] == '2':
                tokenized_text['labels'] = label_maps[example['class2']]
            return tokenized_text
        
        tokenized_datasets = dataset.map(process_function)

        data_collator = CustomCollatorWithPadding(tokenizer=tokenizer)

        def compute_metrics(eval_pred):
            results = {}
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
            f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")
            precision =  metric_precision.compute(predictions=predictions, references=labels, average="macro")
            recall =  metric_recall.compute(predictions=predictions, references=labels, average="macro")
            results.update(accuracy)
            results.update(precision)
            results.update(recall)
            results.update(f1)
            return results
        model = SeperateTaskForClassification(model_name,
                     input_type,
                     classification_type,
                     num_labels,
                     use_focal=args.use_focal,
                     focal_alpha=args.focal_alpha,
                     use_erroneous_span=True)
        metrics = compute_metrics
    # =================== ==================

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=args.test_batch_size_per_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_steps=100000,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if args.do_eval else None,
        compute_metrics=metrics,
        preprocess_logits_for_metrics=pre_metrics if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if args.do_eval:
        logger.info("*** Evaluate ***")
        if args.task_name == 'MiSEWDetection' or args.task_name == 'spandetection':
            metrics = compute_metrics_mapping(trainer, tokenized_datasets['validation'])
        else:
            metrics = trainer.evaluate()
        #metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
    if args.do_test:
        logger.info("*** Evaluate ***")
        if args.task_name == 'MiSEWDetection' or args.task_name == 'spandetection':
            metrics = compute_metrics_mapping(trainer, tokenized_datasets['test'])
        else:
            metrics = trainer.evaluate(tokenized_datasets["test"])
        trainer.log_metrics("test", metrics)


