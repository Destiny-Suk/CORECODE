import json
from pathlib import Path
import random
"""
生成数据，每500条为一批
dict_naturalconv_dialogue-3.json 为减掉 第3批(500条).json 后剩余的对话数据，
同理 dict_dulemon_dialogue-3.json也是
"""

###第2批 --> batch=1
###第3批 --> batch=2
# 后面运行只需每次将batch改为3,4,5,6
batch = 18



all_dialogue = []

# 取300个naturalconv_dialogue.json
def process_na_data():
    with open("data/dict_naturalconv_dialogue-"+str(batch)+".json") as f:
        na_data = json.load(f)
        print('len(na_data)', len(na_data))  #8795
    with open("data/dict_naturalconv_dialogue-"+str(batch+1)+".json", "w") as f_out:
        global all_dialogue 
        all_dialogue += na_data[:300]
        # lens = [len(dia["dialogue"]) for dia in filterd_all_dict]
        # print(lens[:10])
        # json.dump(filterd_all_dict, f_out, ensure_ascii=False)
        f_out.write(json.dumps(na_data[300:], ensure_ascii=False))


# 取100个dulemon_dialogue.json
def process_du_data():
    with open("data/dict_dulemon_dialogue-"+str(batch)+".json") as f:
        du_data = json.load(f)
        print('len(du_data)', len(du_data))  #18196
    with open("data/dict_dulemon_dialogue-"+str(batch+1)+".json", "w") as f_out:
        global all_dialogue 
        all_dialogue += du_data[:200]
        # lens = [len(dia["dialogue"]) for dia in filterd_all_dict]
        # print(lens[:10])
        # json.dump(filterd_all_dict, f_out, ensure_ascii=False)
        f_out.write(json.dumps(du_data[200:], ensure_ascii=False))
    return all_dialogue


if __name__ == "__main__":
    # pathlib的mkdir接收两个参数:
    # parents：如果父目录不存在，是否创建父目录。
    # exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    Path("data/").mkdir(parents=True, exist_ok=True)
    process_na_data()
    all_dialogue = process_du_data()
    with open("data/第"+str(batch+1)+"批(500条).json", "w") as f_out:
        random.shuffle(all_dialogue) #打乱顺序
        f_out.write(json.dumps(all_dialogue, ensure_ascii=False))