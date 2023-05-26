# flake8: noqa
import torch
import argparse
import json
import os
import random
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoModelForCausalLM

import numpy as np
import json, argparse
from tqdm import tqdm
from pathlib import Path

from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from sentence_transformers import SentenceTransformer, util

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tju", default=False, action="store_true",
                        help="Whether to use 9998")

    parser.add_argument("--chat", default=False, action="store_true",
                        help="Whether to use GLM6B-CHAT")
    
    parser.add_argument("--glm_10b_chinese", default=False, action="store_true",
                        help="10B-GLM")
    parser.add_argument("--glm_large_chinese", default=True, action="store_true",
                        help="335m-GLM")
    
    parser.add_argument("--belle_7B_02M", default=False, action="store_true",
                        help="BELLE-SFT-0.2M")
    parser.add_argument("--belle_7B_1M", default=False, action="store_true",
                        help="BELLE-SFT-1M")
    parser.add_argument("--belle_7B_2M", default=False, action="store_true",
                        help="BELLE-SFT-2M")
    
    parser.add_argument("--bloomz_7b1", default=False, action="store_true",
                        help="bloomz_7b1")
    
    parser.add_argument("--use-bminf", default=True, action="store_true",
                       help="Whether to use BMInf")
    parser.add_argument("--memory-limit", type=int, default=4,
                        help="GPU Memory limit, in GB")
    parser.add_argument("--n_shot", type=int, default=4,
                        help="few_shot numbers")
    parser.add_argument("--test", default=False,
                        help="few_shot numbers")
    parser.add_argument("--task_name", type=str, default="", help="task name")
    
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation.")
    parser.add_argument("--input_path", type=str, help="Input json file.")
    parser.add_argument("--output_path", type=str, default="", help="Json file to save generated outputs.")
    args = parser.parse_args()
    return args

def generate_answer(args,text,model,tokenizer):
    if args.chat:
        answer, history = model.chat(tokenizer, text, history=[])
        if args.task_name == 'task1_gen':
            filtered_answer = re.sub(r'[^0-9]','',answer) #将所有的非数字的字符都匹配为''
        elif args.task_name == 'task6_1_gen':
            filtered_answer = answer.split('的原因是')[-1]
    # if args.chat:
    #   answer, history = model.chat(tokenizer, text.replace('[MASK]', ''), history=[])
    elif args.bloomz_7b1:
        print('text:', text)
        inputs = tokenizer.encode(text, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs)
        answer = tokenizer.decode(outputs[0])
        filtered_answer = answer
    elif args.belle_7B_2M or args.belle_7B_02M or args.belle_7B_1M:
        # print(text)
        if args.test:
            input_ids = tokenizer(text.replace('[MASK]', ''), return_tensors="pt").input_ids.cuda()
        else:
            input_ids = tokenizer(text.replace('[MASK]',''), return_tensors="pt").input_ids.cuda()
        outputs = model.generate(input_ids, max_new_tokens=3, do_sample=True, top_k=30, top_p=0.85, temperature=0.35,
                                 repetition_penalty=1.2)
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(answer)
        # answer = str(answer[0]).replace('给出下面问题的正确选项:{}'.format(text), '')
        answer = answer[0].strip().split('正确的选项是')
        answer = str(answer[-1])
        # print("list_answer.{}".format(answer))
        # print("Assistant:\n" + answer[0].strip().replace(text, ""))
        # print("\n------------------------------------------------\nHuman:")
    else: # glm
        # print(text)
        inputs = tokenizer(text, return_tensors="pt")
        # print(inputs)
        inputs = tokenizer.build_inputs_for_generation(inputs+'[MASK]', max_gen_length=512)
        # print(inputs)
        try:
            inputs = {key: value.cuda() for key, value in inputs.items()}
        except:
            return 'E'
        try:
            outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
        except:
            return 'E'
        # outputs_all = tokenizer.decode(outputs[0].tolist())
        # print(tokenizer.decode(outputs[0].tolist()))
        # answer = re.findall(r"<|startofpiece|>(.+?)<|endofpiece|>", outputs_all)
        # print(tokenizer.decode(outputs[0].tolist()))
        # print(tokenizer.decode(outputs[0].tolist()).split(' <|startofpiece|> '))
        try:
            # answer = tokenizer.decode(outputs[0].tolist())
            answer = tokenizer.decode(outputs[0].tolist()).split(' <|startofpiece|> ')[1].split('解析')[0]
        except:
            answer = 'E'
        filtered_answer = re.sub(r'[^0-9]','',answer) #将所有的非数字的字符都匹配为''

    # answer = filter_answer(answer)
    # print("list_1_answer.{}".format(answer))
    # filtered_answer = re.sub(r'[^0-9]','',answer) #将所有的非数字的字符都匹配为''
    # if len(filtered_answer) > 1:
    #     filtered_answer = filtered_answer[0]
    return answer, filtered_answer

# def metrics(gold, generated, scorers, simcse):
def metrics(gold, generated, scorers):
    """
    Compute metrics.
    """
    refs, hyps, task_scores = {}, {}, []
    for j in range(len(gold)):
        refs[j] = [gold[j]]
        hyps[j] = [generated[j]]

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hyps)
        if type(score) == list:
            for m, s in zip(method, score):
                task_scores.append(round(s, 4))
        else:
            task_scores.append(round(score, 4))

    # embeddings1 = simcse.encode(gold, convert_to_tensor=True)
    # embeddings2 = simcse.encode(generated, convert_to_tensor=True)
    # cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    # similarity = round(np.mean(cosine_scores.diag().cpu().numpy()), 4)
    # task_scores.append(similarity)
    return task_scores

if __name__ == "__main__":
    args = get_args()
    # if args.use_tju:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    #     path = 'experiments/chatglm-few-shot/'
    #     dev_path = 'experiments/chatglm-few-shot/'
    # else:
    #     path = 'datammm_untest/'
    #     dev_path = 'datammm_dev/'

    ## Read input json and specify output path ##
    data = [json.loads(line) for line in open(args.input_path).readlines()]
    inputs = [instance["input"] for instance in data]
    answers = [instance["output"] for instance in data]
    
    # args.input_path: 'dataset/generation/all_task1_gen.json'
    output_path = args.output_path
    if output_path == "":
        output_path = "experiments/results/output_" + args.input_path.split("/")[-1].split(".")[0]
    if args.task_name == "":
        args.task_name = args.input_path.split("/")[-1].split(".")[0].split("l_")[1]  #'task1_gen'
    #chatglm
    if args.chat:
        if args.use_tju:
            model_path = "models/chatglm-6b"  # You can modify the path for storing the local model
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        else:
            model_path = "models/chatglm-6b"  # You can modify the path for storing the local model
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        directory = "chatglm-6b"
    else:
        if args.glm_10b_chinese:
            model_path = "models/glm-10b-chinese"  # You can modify the path for storing the local model
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
            model = model.half().cuda()
            directory = "glm-10b-chinese"
        elif args.glm_large_chinese:
            # 335M
            model_path = "models/glm-large-chinese"  # You can modify the path for storing the local model
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
            model = model.cuda()
            directory = "glm-large-chinese"
        elif args.belle_7B_2M:
            model_path = "models/BELLE-7B-2M"  # You can modify the path for storing the local model
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = model.half().cuda()
            directory = "BELLE-7B-2M"
        elif args.belle_7B_02M:
            model_path = "models/BELLE-7B-0.2M"  # You can modify the path for storing the local model
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = model.half().cuda()
            directory = "BELLE-7B-0.2M"
        elif args.belle_7B_1M:
            model_path = "models/BELLE-7B-1M"  # You can modify the path for storing the local model
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = model.half().cuda()
            directory = "BELLE-7B-1M"

        elif args.bloomz_7b1:
            model_path = "models/bloomz-7b1"
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            directory = "bloomz-7b1"

    each_model_output_path = "/".join((output_path, directory)) #each_model_output_path:
    Path(each_model_output_path).mkdir(parents=True, exist_ok=True)
    

    corrects = []
    correct_count = 0
    n=0
    output_path = '{}/{}-shot-output.txt'.format(each_model_output_path, n)
    metrics_path = '{}/{}-shot-metrics.txt'.format(each_model_output_path, n)
    # if os.path.exists(output_path):
    #     raise ValueError(f'Output directory ({output_path}) already exists and is not empty.')
    predictions = []
    with open(output_path,'a') as f:
        num = 10 #前num条
        # for i,question in enumerate(inputs):
        for i,question in enumerate(inputs[:num]):
            full_prediction, prediction = generate_answer(args, question, model, tokenizer)
            predictions.append(full_prediction)
            # 将前100条的问题答案写到文件里
            if i <= 100:
                print("--- 问题 ---:{}".format(question))
                print("--- 正确答案 ---:{}".format(answers[i]))
                # print("--- 生成答案 ---:{}".format(prediction))
                print("--- 带解析答案 ---:{}".format(full_prediction))
                f.write("\n".join(("--- 问题 ---:{}".format(question),
                        "--- 正确答案 ---:{}".format(answers[i]),
                        "--- 生成答案 ---:{}".format(prediction),
                        "--- 带解析答案 ---:{}".format(full_prediction))))
                f.write("\n\n")

    ## Scorers and Sentence emebedding model ##
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    # simcse = SentenceTransformer("princeton-nlp/sup-simcse-roberta-large").cuda()

    # scores = metrics(answers, predictions, scorers, simcse)
    scores = metrics(answers[:num], predictions[:num], scorers)

    results = []
    # metric_name = ["BLEU1", "BLEU2", "BLEU3", "BLEU4", "METEOR", "ROUGE_L", "CIDER", "Sem-Sim"]
    metric_name = ["BLEU1", "BLEU2", "BLEU3", "BLEU4", "METEOR", "ROUGE_L", "CIDER"]
    for m, s in zip(metric_name, scores):
        results.append(m + ": " + str(s))
    print ("Metrics:")
    print ("\n".join(results))

    with open(metrics_path, "a") as f:
        f.write("\n".join(results) + "\n\n")
    
    print ("Saved results in {}.".format(each_model_output_path))
