import json
import glob
import re
from pathlib import Path

commonsense_type = ['entity', 'event', 'social_interaction']


# 处理对话数据，处理为格式：每条标注为1行
def process_data(files):
    """
    把所有的原因：事件原因域和事件：后续事件域的标注数据提取出来
    """
    with open("data/all.json", "w") as f_out:
        all_len_dia, all_len_anno = 0, 0
        for file in files:
            start_id, end_id = re.findall(r"[(](.*?)[)]", file)[0].split('-')[0], re.findall(r"[(](.*?)[)]", file)[0].split('-')[1]
            with open(file) as f:
                dialogue_ids, annotation_count = set(), 0
                data = json.load(f)
                for dialogue in data[int(start_id)-1: int(end_id)]:  # dialogue: dict
                    dialogue_id = dialogue["dialogue_id"]
                    dialogue_content = dialogue["dialogue"]
                    identified_dialogue_content = ['A: ' + dialogue_content[i] if i % 2 == 0 else 'B: ' + dialogue_content[i]
                                            for i in range(len(dialogue_content))]
                    if len(dialogue["commonsense_type"]["entity"])+len(dialogue["commonsense_type"]["event"])+len(dialogue["commonsense_type"]["social_interaction"]) > 0: # 不全为[]
                        dialogue_ids.add(dialogue_id)
                    for type in commonsense_type: # type: 'entity', 'event', 'social_interaction'
                        instances = list(dialogue[type].keys()) # ['我的拳王男友', '杜琪峰', '向佐']
                        for instance_name in instances:
                            annotations = dialogue[type][instance_name]["annotation"]  # list
                            for anno_id, annotation in enumerate(annotations):
                                domin = annotation.split(' = ')[0].split(': ')[0]
                                slot = annotation.split(' = ')[0].split(': ')[1]
                                value = annotation.split(' = ')[1]
                                texts = dialogue[type][instance_name]["text_group"][anno_id]
                                texts_ids = dialogue[type][instance_name]["text_span_index"][anno_id] #对话中原文文本的起始和终止位置
                                text_utt_indexes = dialogue[type][instance_name]["text_utterance_index"][anno_id] #对话中原文文本的utterance索引(从0开始)
                                conflicts = dialogue[type][instance_name]["conflict_candidate"][anno_id]
                                line = {type+"_name": instance_name, "domin": domin, "slot": slot, "value": value,
                                        "texts": texts, "texts_ids": texts_ids, "text_utt_indexes": text_utt_indexes, "conflicts": conflicts, 
                                        "dialogue_id": dialogue_id, "dialogue_content": identified_dialogue_content}
                                f_out.write(json.dumps(line, ensure_ascii=False) + "\n")
                                annotation_count += 1
                print("="*5, file, "="*5)
                print("{}条对话存在标注，共{}条标注".format(len(dialogue_ids), annotation_count))
                all_len_dia += len(dialogue_ids)
                all_len_anno += annotation_count
    return all_len_dia, all_len_anno
    
    
if __name__ == "__main__":
    # pathlib的mkdir接收两个参数:
    # parents：如果父目录不存在，是否创建父目录。
    # exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    Path("dataset/").mkdir(parents=True, exist_ok=True)
    # for split in ["train", "val", "test"]:
    json_files = glob.glob("data/raw_data/*.json")
    len_dia, len_anno = process_data(json_files)
    # for file in json_files:
    #     process_data(file)
    print("There is {} annotations from {} dialogues".format(len_anno, len_dia))