import json
from pathlib import Path
import random
"""
展示某批数据的对话数量和对话id
"""


with open("data/第1批(500条).json") as f:
    data = json.load(f)
    print('len(data)', len(data))  # 500
    ids = []
    ids =[dialogue["dialogue_id"] for dialogue in data]
    print("ids: ",ids)