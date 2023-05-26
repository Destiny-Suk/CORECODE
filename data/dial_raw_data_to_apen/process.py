import json
from pathlib import Path
# 本处理过滤了'样例数据.json'中的对话，并将对话内容变成了下面格式
# {"P1": xxxx}
# {"P2": xxxx}
# 并且按对话轮数从多到少排序

# 按id过滤对话数据
dial_count = 0

with open("dulemon_dialogue.json") as f:
    du_data = json.load(f)
print(len(du_data))  #18346

with open("naturalconv_dialogue.json") as f:
    na_data = json.load(f)
print(len(na_data))  #9145

with open("样例数据.json") as f:
    fi_data = json.load(f)
print(len(fi_data))  #100

example_dial_id_list = []
for dial in fi_data:
    example_dial_id_list.append(dial["dialogue_id"])
print('dial_id_list:', example_dial_id_list)


# 处理naturalconv_dialogue.json
def process_na_data():
    filterd_all_dict = []
    with open("data/dict_naturalconv_dialogue.json", "w") as f_out:
        for dialogue in na_data:  # dialogue: dict
            if dialogue["dialogue_id"] not in example_dial_id_list:
                dialogue_content = dialogue["dialogue"]
                new_dialogue_content = []
                for utt_id, dialogue_utt in enumerate(dialogue_content):
                    if utt_id%2==0:
                        new_dialogue_content.append({"P1":dialogue_utt})
                    else:
                        new_dialogue_content.append({"P2":dialogue_utt})
                new_dialogue = {"dialogue_id": dialogue["dialogue_id"],
                                "dialogue": new_dialogue_content,
                                "concepts": dialogue["concepts"]
                                }
                filterd_all_dict.append(new_dialogue)
        lens = [len(dia["dialogue"]) for dia in filterd_all_dict]
        print(lens[:10])
        # json.dump(filterd_all_dict, f_out, ensure_ascii=False)
        f_out.write(json.dumps(filterd_all_dict, ensure_ascii=False))


# 处理dulemon_dialogue.json
def process_du_data():
    filterd_all_dict = []
    with open("data/dict_dulemon_dialogue.json", "w") as f_out:
        for dialogue in du_data:  # dialogue: dict
            if dialogue["dialogue_id"] not in example_dial_id_list:
                dialogue_content = dialogue["dialogue"]
                new_dialogue_content = []
                for utt_id, dialogue_utt in enumerate(dialogue_content):
                    if utt_id%2==0:
                        new_dialogue_content.append({"P1":dialogue_utt})
                    else:
                        new_dialogue_content.append({"P2":dialogue_utt})
                new_dialogue = {"dialogue_id": dialogue["dialogue_id"],
                                "dialogue": new_dialogue_content,
                                "concepts": dialogue["concepts"]
                                }
                filterd_all_dict.append(new_dialogue)
        sorted(filterd_all_dict, key = lambda x: len(x["dialogue"]))
        lens = [len(dia["dialogue"]) for dia in filterd_all_dict]
        print(lens[:10])
        # json.dump(filterd_all_dict, f_out, ensure_ascii=False)
        f_out.write(json.dumps(filterd_all_dict, ensure_ascii=False))


    

if __name__ == "__main__":
    # pathlib的mkdir接收两个参数:
    # parents：如果父目录不存在，是否创建父目录。
    # exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    Path("data/").mkdir(parents=True, exist_ok=True)
    process_du_data()
    process_na_data()