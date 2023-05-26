import json
from pathlib import Path

batch = 0
all_dialogue = []

# 取样例数据.json的
def process_exa_data():
    with open("样例数据.json") as f:
        exa_data = json.load(f)
        print('len(ex_data)', len(exa_data))  #9095
        global all_dialogue 
        all_dialogue += exa_data
    
# 取400条.json的
def process_400_data():
    with open("data/第1批(400条).json") as f:
        exa_data = json.load(f)
        print('len(ex_data)', len(exa_data))  #9095
        global all_dialogue 
        all_dialogue += exa_data
    return all_dialogue

if __name__ == "__main__":
    # pathlib的mkdir接收两个参数:
    # parents：如果父目录不存在，是否创建父目录。
    # exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    Path("data/").mkdir(parents=True, exist_ok=True)
    process_exa_data()
    all_dialogue = process_400_data()
    with open("data/第"+str(batch+1)+"批(500条).json", "w") as f_out:
        # random.shuffle(all_dialogue) #打乱顺序
        f_out.write(json.dumps(all_dialogue, ensure_ascii=False))
