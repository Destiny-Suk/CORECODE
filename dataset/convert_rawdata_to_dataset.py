import argparse
import copy
import json
import glob
import random
import re
import copy
import tqdm
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# 常识推理 三选一
# 对话常识领域识别 n选一
# 对话常识类型识别 n选一
entity_domins = {
    '属性':0,
    '比较':1,
    '空间':2
}

event_domins = {
    '前提':0,
    '原因':1,
    '后续':2,
    '时间':3,
    '空间':4
}
social_interaction_domins = {}

entity_slots = {
    '是':0, '是一个':1, '有':2, '包括...类型':3, '由...制成':4, '是...的一部分':5, '用途':6, '能力':7,
    '相同物':8, '相似物':9, '相反物':10,
    '在某个位置':11, '空间上相邻':12, '空间上包含':13
}

event_slots = {
    '前提条件':0,
    '事件原因':1, '情绪原因':2, '时间原因':3, '空间位置原因':4,
    '后续事件':5, '后续情绪反应':6, '后续时间改变':7, '后续空间位置改变':8,
    '发生时间':9, '开始时间':10, '结束时间':11, '持续时长':12, '频率':13,
    '发生位置':14
}

social_interaction_slots = {
    'x的属性':0, 'x应该':1, 'x的意图':2, 'x的反应':3, '对x的影响':4, 'y的反应':5, '对y的影响':6
}

matching_dict = {
    "entity_name": [[entity_domins, entity_slots], "源实体", "目标实体"],
    "event_name": [[event_domins, event_slots], "源事件", "目标事件"],
    "social_interaction": [[social_interaction_domins, social_interaction_slots], "源社交", "目标社交"]
}
q1 = [
    "What is or could be the cause of target?",
    "What is or could be the prerequisite of target?",
    "What is the possible emotional reaction of the listener in response to target?"
]
q2 = [
    "What is or could be the motivation of target?",
    "What subsequent event happens or could happen following the target?"
]


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/all.json')
    return parser.parse_args()

def read_json(file_path):
    results = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            results.append(json.loads(line))
    f.close()
    return results

def createbenchmark1(dataset, save_path):
    nums = 0
    for data in dataset:
        if data['正错判断'] == '正确':
            nums += 1
    print(nums, nums / len(dataset) * 100)  # 正确的句子所占的比例
    with open(save_path, 'w') as f:
        for d in dataset:
            example = {'text':d['原文'], 'label':d['正错判断']}
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    f.close()

def createbenchmark4(dataset, save_path):
    results = []
    for d in dataset:
        example = {}
        raw_text = d['原文']
        for de in d['具体错误']:
            span = [int(x) for x in de['错误位置'][1:-1].split(',')]
            label = 0
            txt = []
            span = [span[0], span[1]] if span[0] <= span[1] else [span[1], span[0]] # 错误span位置，[11,12]
            if raw_text[0:span[0]] != '': # 错误span前面有没有文字，有的话把文字加到list里，且label为1；没有的话为0
                txt.append(raw_text[0:span[0]])
                label += 1
            txt.append(raw_text[span[0]:span[1]+1])
            txt.append(raw_text[span[1]+1:]) # [错误span前面的文字，错误span，错误span后面的文字]
            example['text'] = ' '.join(txt)
            example['span'] = label  # 表示错误span前面有没有文字，有的话为1；没有的话为0
            example['class1'] = de['错误大类']
            example['class2'] = de['错误小类']
            #example['detail'] = d
            results.append(copy.deepcopy(example))
    print("{} have total {}".format(save_path, len(results)))
    with open(save_path, 'w') as f:
        for example in results:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    f.close()

# 选择式常识推理 三选一
# selective_commonsense_reasoning
def task1_cla(split, zero_shot=False):
    data = [json.loads(line) for line in open("data/" + split + ".json").readlines()]
    sep = " \\n "
    f = open("dataset/classification/" + split + "_task1_cla.json", "w")
    for instance in data:
        original_phrase = list(instance["conflicts"].keys())[0]
        phrase_id = instance["texts"].index(original_phrase)
        start, end = instance["texts_ids"][phrase_id][0], instance["texts_ids"][phrase_id][1]
        tmp_dialogue_content = '\n\n'.join(instance["dialogue_content"])
        dialogue_content = tmp_dialogue_content[:start]+"<MASK>"+tmp_dialogue_content[end:]
        dialogue_utts = dialogue_content.split('\n\n')
        c = " <utt> ".join(dialogue_utts)  #带mask

        choices = [original_phrase]
        for conflict_phrase in instance["conflicts"][original_phrase]:
            choices.append(conflict_phrase)
            random.shuffle(choices) #打乱顺序
        context = sep.join(["对话中的<MASK>处应填入什么？", "对话: " + c])
        line = {
            "ID": instance["dialogue_id"], "context": context, "choice0": choices[0], "choice1": choices[1], "choice2": choices[2], 
            "label": choices.index(original_phrase)
        }
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    f.close()

# 生成式常识推理
# generative_commonsense_reasoning
def task1_gen(split, zero_shot=False):
    data = [json.loads(line) for line in open("data/" + split + ".json").readlines()]
    sep = " \\n "
    
    # 对于zero_shot实验，用原因、前提、情绪反应任务做训练和验证，在动机、后续任务上测试
    if zero_shot:
        f = open("dataset/generation/" + split + "_zs_generative_commonsense_reasoning.json", "w")
        if split == "test":
            question_set = q2
        else:
            question_set = q1
    # 对于非zero_shot实验，在所有5个任务上训练或测试
    else:
        f = open("dataset/generation/" + split + "_task1_gen.json", "w")
        question_set = q1 + q2    
    for instance in data:
        original_phrase = list(instance["conflicts"].keys())[0]
        phrase_id = instance["texts"].index(original_phrase)
        start, end = instance["texts_ids"][phrase_id][0], instance["texts_ids"][phrase_id][1]
        tmp_dialogue_content = '\n\n'.join(instance["dialogue_content"])
        dialogue_content = tmp_dialogue_content[:start]+"<MASK>"+tmp_dialogue_content[end:]
        dialogue_utts = dialogue_content.split('\n\n')
        c = " <utt> ".join(dialogue_utts)  #带mask

        choices, choice_str = [], ""
        choices.append(original_phrase)
        for conflict_phrase in instance["conflicts"][original_phrase]:
            choices.append(conflict_phrase)
            random.shuffle(choices) #打乱顺序
        for k, num in enumerate(["(0)", "(1)", "(2)"]):
            choice_str += num + " " + choices[k] + " "
        choice_str = choice_str[:-1]
        context = sep.join(["对话中的<MASK>处应填入什么？", choice_str, "对话: " + c])
        answer = original_phrase
        line = {"input": context, "output": answer}
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    f.close()

# 对话常识冲突检测 抽取式问答
def task3(split):
    data = [json.loads(line) for line in open("data/" + split + ".json").readlines()]
    sep = " \\n "
    f = open("dataset/span_extraction/" + split + "_task3.json", "w")
    idx = 0
    for instance in data:
        original_phrase = list(instance["conflicts"].keys())[0]
        phrase_id = instance["texts"].index(original_phrase)
        start, end = instance["texts_ids"][phrase_id][0], instance["texts_ids"][phrase_id][1]
        tmp_dialogue_content = '\n\n'.join(instance["dialogue_content"])
        for i in range(len(instance["conflicts"][original_phrase])):  # 2
            # 使用标注的常识冲突短语替换原文中的相应短语
            dialogue_content = tmp_dialogue_content[:start] + instance["conflicts"][original_phrase][i] + tmp_dialogue_content[end:]
            dialogue_utts = dialogue_content.split('\n\n')
            c = " <utt> ".join(dialogue_utts)  #带mask
            q = "对话中哪里存在常识错误？"
            answers = {
                "text": [instance["conflicts"][original_phrase][i]], 
                "answer_start": [start]
                }
            example = {
                "id": instance["dialogue_id"]+'_'+str(idx), 
                "context": c,
                "question": q, 
                "answers": answers
                }
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
            idx += 1
    f.close()

# 选择式对话常识领域识别 n选一
# selective_domain_identification
def task5_cla(split, zero_shot=False):
    data = [json.loads(line) for line in open("data/" + split + ".json").readlines()]
    sep = " \\n "
    f = open("dataset/classification/" + split + "_task5_cla.json", "w")
    for instance in data:
        if 'entity_name' in list(instance.keys()):
            choices = list(entity_domins.keys())
            c = " <utt> ".join(instance["dialogue_content"])
            context = sep.join(["源实体和目标实体的关系属于哪一个领域？", "源实体: " + instance["entity_name"], "目标实体: " + instance["value"], "对话上下文: " + c])
            line = {
                "ID": instance["dialogue_id"], "context": context, "choice0": choices[0], "choice1": choices[1], "choice2": choices[2], 
                "label": entity_domins[instance["domin"]]
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        if 'event_name' in list(instance.keys()):
            choices = list(event_domins.keys())
            c = " <utt> ".join(instance["dialogue_content"])
            context = sep.join(["源事件和目标事件的关系属于哪一个领域？", "源事件: " + instance["event_name"], "目标事件: " + instance["value"], "对话上下文: " + c])
            line = {
                "ID": instance["dialogue_id"], "context": context, "choice0": choices[0], "choice1": choices[1], "choice2": choices[2], "choice3": choices[3], "choice4": choices[4],
                "label": event_domins[instance["domin"]]
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        # 社交只有一个域，无
    f.close()

# 生成式对话常识领域识别
# generative_domain_identification
def task5_gen(split, zero_shot=False):
    data = [json.loads(line) for line in open("data/" + split + ".json").readlines()]
    sep = " \\n "
    if zero_shot:
        f = open("dataset/generation/" + split + "_zs_generative_domain_identification.json", "w")
        if split == "test":
            question_set = q2
        else:
            question_set = q1
    else:
        f = open("dataset/generation/" + split + "_task5_gen.json", "w")
        question_set = q1 + q2
    for instance in data:
        type_name = list(instance.keys())[0]  # "entity_name" / "event_name"
        print(type_name)
        domins = matching_dict[type_name][0][0]
        print('domins', domins)
        source = matching_dict[type_name][1]
        target = matching_dict[type_name][2]
        choices, choice_str = list(domins.keys()), ""
        if 'entity_name' in list(instance.keys()):
            for k, num in enumerate(["(0)", "(1)", "(2)"]):
                choice_str += num + " " + choices[k] + " "
            choice_str = choice_str[:-1]
        elif 'event_name' in list(instance.keys()):
            for k, num in enumerate(["(0)", "(1)", "(2)", "(3)", "(4)"]):
                choice_str += num + " " + choices[k] + " "
            choice_str = choice_str[:-1]
        c = " <utt> ".join(instance["dialogue_content"])
        context = sep.join([source+"和"+target+"的关系属于哪一个领域？", source+": " + instance[type_name], target+": " + instance["value"], "对话上下文: " + c])
        answer = instance["domin"]
        line = {"input": context, "output": answer}
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    f.close()

# 选择式对话常识类型识别 n选一
# selective_slot_identification
def task6_cla(split, zero_shot=False):
    data = [json.loads(line) for line in open("data/" + split + ".json").readlines()]
    sep = " \\n "
    f = open("dataset/classification/" + split + "_task6_cla.json", "w")
    for instance in data:
        if 'entity_name' in list(instance.keys()):
            choices = list(entity_slots.keys())
            c = " <utt> ".join(instance["dialogue_content"])
            context = sep.join(["源实体和目标实体的关系属于哪一个类型？", "源实体: " + instance["entity_name"], "目标实体: " + instance["value"], "对话上下文: " + c])
            line = {
                "ID": instance["dialogue_id"], "context": context, 
                "choice0": choices[0], "choice1": choices[1], "choice2": choices[2], "choice3": choices[3], "choice4": choices[4],
                "choice5": choices[5], "choice6": choices[6], "choice7": choices[7], "choice8": choices[8], "choice9": choices[9],
                "choice10": choices[10], "choice11": choices[11], "choice12": choices[12], "choice13": choices[13],
                "label": entity_slots[instance["slot"]]
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        if 'event_name' in list(instance.keys()):
            choices = list(event_slots.keys())
            c = " <utt> ".join(instance["dialogue_content"])
            context = sep.join(["源事件和目标事件的关系属于哪一个类型？", "源事件: " + instance["event_name"], "目标事件: " + instance["value"], "对话上下文: " + c])
            line = {
                "ID": instance["dialogue_id"], "context": context, 
                "choice0": choices[0], "choice1": choices[1], "choice2": choices[2], "choice3": choices[3], "choice4": choices[4],
                "choice5": choices[5], "choice6": choices[6], "choice7": choices[7], "choice8": choices[8], "choice9": choices[9],
                "choice10": choices[10], "choice11": choices[11], "choice12": choices[12], "choice13": choices[13],
                "label": event_slots[instance["slot"]]
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        if 'social_interaction_name' in list(instance.keys()):
            choices = list(social_interaction_slots.keys())
            c = " <utt> ".join(instance["dialogue_content"])
            context = sep.join(["源社交和目标社交的关系属于哪一个类型？", "源社交: " + instance["social_interaction_name"], "目标社交: " + instance["value"], "对话上下文: " + c])
            line = {
                "ID": instance["dialogue_id"], "context": context, 
                "choice0": choices[0], "choice1": choices[1], "choice2": choices[2], "choice3": choices[3], "choice4": choices[4],
                "choice5": choices[5], "choice6": choices[6],
                "label": social_interaction_slots[instance["slot"]]
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    f.close()

# 生成式对话常识类型识别
# generative_slot_identification
def task6_gen(split, zero_shot=False):
    data = [json.loads(line) for line in open("data/" + split + ".json").readlines()]
    sep = " \\n "
    if zero_shot:
        f = open("dataset/generation/" + split + "_zs_generative_slot_identification.json", "w")
        if split == "test":
            question_set = q2
        else:
            question_set = q1
    else:
        f = open("dataset/generation/" + split + "_task6_gen.json", "w")
        question_set = q1 + q2
    for instance in data:
        type_name = list(instance.keys())[0]  # "entity_name" / "event_name"
        print(type_name)
        slots = matching_dict[type_name][0][1]
        print('slots', slots)
        source = matching_dict[type_name][1]
        target = matching_dict[type_name][2]
        choices, choice_str = list(slots.keys()), ""
        print("choices in generative_slot_identification()", choices)
        if 'entity_name' in list(instance.keys()):
            for k, num in enumerate(["(0)", "(1)", "(2)", "(3)", "(4)","(5)", "(6)", "(7)", "(8)", "(9)","(10)", "(11)", "(12)", "(13)"]):
                choice_str += num + " " + choices[k] + " "
            choice_str = choice_str[:-1]
        elif 'event_name' in list(instance.keys()):
            for k, num in enumerate(["(0)", "(1)", "(2)", "(3)", "(4)","(5)", "(6)", "(7)", "(8)", "(9)","(10)", "(11)", "(12)", "(13)"]):
                choice_str += num + " " + choices[k] + " "
            choice_str = choice_str[:-1]
        elif 'social_interaction_name' in list(instance.keys()):
            for k, num in enumerate(["(0)", "(1)", "(2)", "(3)", "(4)","(5)", "(6)"]):
                choice_str += num + " " + choices[k] + " "
            choice_str = choice_str[:-1]
        c = " <utt> ".join(instance["dialogue_content"])
        context = sep.join([source+"和"+target+"的关系属于哪一个类型？", source+": " + instance[type_name], target+": " + instance["value"], "对话上下文: " + c])
        answer = instance["slot"]
        line = {"input": context, "output": answer}
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    f.close()

def split_to_datasets():
    for task_dir in ["classification","generation","span_extraction"]:
        json_files = glob.glob("dataset/"+task_dir+"/all_*.json")  #'dataset/span_extraction/all_task3.json'
        for file in json_files:
            task_name = file.split("/")[-1].split("ll_")[1].split(".")[0] #'task3'
            f_train = open(os.path.join("dataset/",task_dir,task_name +"_11111_train.json"), "w")
            f_dev = open(os.path.join("dataset/",task_dir,task_name +"_11111_dev.json"), "w")
            f_test = open(os.path.join("dataset/",task_dir,task_name +"_11111_test.json"), "w")
            data = [json.loads(line) for line in open(file).readlines()]
            train_data, dev_test_data = train_test_split(data, test_size=0.4, shuffle=True)
            dev_data, test_data = train_test_split(dev_test_data, test_size=0.5, shuffle=True)
            print(task_name+' train set size: ', len(train_data))
            print(task_name+' valid set size: ', len(dev_data))
            print(task_name+' test set size: ', len(test_data))
            for data in train_data:
                f_train.write(json.dumps(data, ensure_ascii=False) + "\n")
            for data in dev_data:
                f_dev.write(json.dumps(data, ensure_ascii=False) + "\n")
            for data in test_data:
                f_test.write(json.dumps(data, ensure_ascii=False) + "\n")
            f_train.close()
            f_dev.close()
            f_test.close()
                
def main(args):
    # dataset_all = read_json(args.data_path)

    Path("dataset/classification/").mkdir(parents=True, exist_ok=True)
    Path("dataset/generation/").mkdir(parents=True, exist_ok=True)
    Path("dataset/span_extraction/").mkdir(parents=True, exist_ok=True)
    
    # for split in ["train", "val", "test"]:
    split = "all"
    
    task3(split)
    task1_cla(split)
    task1_gen(split)
    task5_cla(split)
    task5_gen(split)
    task6_cla(split)
    task6_gen(split)
    split_to_datasets()

    # # Benchmark 1
    # createbenchmark1(train_dataset, './benchmark1/train.json')
    # createbenchmark1(dev_dataset, './benchmark1/dev.json')
    # createbenchmark1(test_dataset,'./benchmark1/test.json')

    # # Benchmark 2
    # createbenchmark2(train_dataset_error, './benchmark2/train.json')
    # createbenchmark2(dev_dataset_error, './benchmark2/dev.json')
    # createbenchmark2(test_dataset_error,'./benchmark2/test.json')

    # # Benchmark 3
    # print("Train Example have:{}".format(len(train_dataset_error)))
    # print("Dev Example have:{}".format(len(dev_dataset_error)))
    # print("Test Example have:{}".format(len(test_dataset_error)))
    # createbenchmark3_misewbased(train_dataset_error, './benchmark3/train.json')
    # createbenchmark3_misewbased(dev_dataset_error, './benchmark3/dev.json')
    # createbenchmark3_misewbased(test_dataset_error,'./benchmark3/test.json')

    # # Benchmark 4
    # print("Train Example have:{}".format(len(train_dataset_error)))
    # print("Dev Example have:{}".format(len(dev_dataset_error)))
    # print("Test Example have:{}".format(len(test_dataset_error)))
    # createbenchmark4(train_dataset_error, './benchmark4/train.json')
    # createbenchmark4(dev_dataset_error, './benchmark4/dev.json')
    # createbenchmark4(test_dataset_error,'./benchmark4/test.json')

    # # Benchmark 5
    # print("Train Example have:{}".format(len(train_dataset_error)))
    # print("Dev Example have:{}".format(len(dev_dataset_error)))
    # print("Test Example have:{}".format(len(test_dataset_error)))
    # createbenchmark5(train_dataset_error, './benchmark5/train.json')
    # createbenchmark5(dev_dataset_error, './benchmark5/dev.json')
    # createbenchmark5(test_dataset_error,'./benchmark5/test.json')

    # # Benchmark 6s
    # #train_ben6, dev_ben6, test_ben6 = split_data(error_text, train_rate, dev_rate)
    # createbenchmark6(train_dataset_error, './benchmark6/train.json')
    # createbenchmark6(dev_dataset_error, './benchmark6/dev.json')
    # createbenchmark6(test_dataset_error,'./benchmark6/test.json')


if __name__ == '__main__':
    parse = parse_config()
    main(parse)