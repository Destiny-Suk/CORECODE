# flake8: noqa
import torch
import argparse
import json
# import xlsxwriter as xw
import xlrd
import os
import random
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel,AutoModelForCausalLM


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tju", default=False, action="store_true",
                        help="Whether to use 9998")

    parser.add_argument("--chat", default=False, action="store_true",
                        help="Whether to use GLM6B-CHAT")
    
    parser.add_argument("--use_10b", default=False, action="store_true",
                        help="10B-GLM")
    parser.add_argument("--use_335m", default=True, action="store_true",
                        help="335m-GLM")
    
    parser.add_argument("--belle_7B_02M", default=False, action="store_true",
                        help="BELLE-SFT-0.2M")
    parser.add_argument("--belle_7B_1M", default=False, action="store_true",
                        help="BELLE-SFT-1M")
    parser.add_argument("--belle_7B_2M", default=False, action="store_true",
                        help="BELLE-SFT-2M")

    parser.add_argument("--use-bminf", default=True, action="store_true",
                       help="Whether to use BMInf")
    parser.add_argument("--memory-limit", type=int, default=4,
                        help="GPU Memory limit, in GB")
    parser.add_argument("--n_shot", type=int, default=4,
                        help="few_shot numbers")
    parser.add_argument("--test", default=False,
                        help="few_shot numbers")
    args = parser.parse_args()
    return args

def filter_answer(answer):
    max_len = 0
    final_ans = ''
    for ans in answer:
        if len(ans) > max_len:
            final_ans = ans
            max_len = len(ans)
    return final_ans

def get_text(args,text,model,tokenizer):
    if args.chat:
        if args.test:
            answer, history = model.chat(tokenizer, text.replace('[MASK]', ''), history=[])
        else:
            answer, history = model.chat(tokenizer, text.replace('[MASK]',''), history=[])
    elif args.belle_7B_2M or args.belle_7B_02M or args.belle_7B_1M:
        # print(text)
        if args.test:
            input_ids = tokenizer(text.replace('[MASK]', ''), return_tensors="pt").input_ids.cuda()
        else:
            input_ids = tokenizer(text.replace('[MASK]',''), return_tensors="pt").input_ids.cuda()
        outputs = model.generate(input_ids, max_new_tokens=3, do_sample=True, top_k=30, top_p=0.85, temperature=0.35,
                                 repetition_penalty=1.2)
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(answer)
        # answer = str(answer[0]).replace('给出下面问题的正确选项:{}'.format(text), '')
        answer = answer[0].strip().split('正确的选项是')
        answer = str(answer[-1])
        # print("list_answer.{}".format(answer))
        # print("Assistant:\n" + answer[0].strip().replace(text, ""))
        # print("\n------------------------------------------------\nHuman:")

    else:
        # glm
        # args = get_args()
        # print(text)
        inputs = tokenizer(text, return_tensors="pt")
        # print(inputs)
        inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
        # print(inputs)
        try:
            inputs = {key: value.cuda() for key, value in inputs.items()}
        except:
            return 'E'
        try:
            outputs = model.generate(**inputs, max_length=3, eos_token_id=tokenizer.eop_token_id)
        except:
            return 'E'
        # outputs_all = tokenizer.decode(outputs[0].tolist())
        # print(tokenizer.decode(outputs[0].tolist()))
        # answer = re.findall(r"<|startofpiece|>(.+?)<|endofpiece|>", outputs_all)
        # print(tokenizer.decode(outputs[0].tolist()))
        # print(tokenizer.decode(outputs[0].tolist()).split(' <|startofpiece|> '))
        try:
            answer = tokenizer.decode(outputs[0].tolist()).split(' <|startofpiece|> ')[1]
        except:
            answer = 'E'
    # answer = filter_answer(answer)
    # print("list_1_answer.{}".format(answer))
    random_output = re.sub(r'[^ABCD]','',answer)
    # if len(random_output) > 1:
    #     random_output = random_output[0]
    # print(random_output)
    return random_output


def read_5shot(pathfile,choices,n): # n: n-shot,即0,1,2,3,4,5
    demostrations = ''
    excel = xlrd.open_workbook(pathfile)
    sheet_names = excel.sheet_names()
    sheet1 = excel.sheet_by_name(sheet_names[0])
    nrows = sheet1.nrows
    ncols = sheet1.ncols
    for i in range(nrows):
        if i<n:
            question = ''
            for col in range(ncols-1):
                if col <= 3:
                    # print(sheet1.cell_value(i,col))
                    question += str(sheet1.cell_value(i, col)) + '({})'.format(choices[col])
                else:
                    question += str(sheet1.cell_value(i, col))+ '。'+ '正确的选项是:'
            # question+='答案是:[gMASK]'
            question += sheet1.cell_value(i, ncols-1) + '。'
            # print(question)
            demostrations += question
        else:
            break

    return demostrations


def read_excel(pathfile,pathfile_dev,n):
    questions,answers = [],[] 
    choices = ['A','B','C','D']
    # 实例化Excel对象
    excel = xlrd.open_workbook(pathfile)  #path_file: 'experiments/chatglm-few-shot/艺术类之绘画.xlsx'
    # 获取Excel中所有的sheet名
    sheet_names = excel.sheet_names()
    # print("sheet_names", excel.sheet_names())
    # 将一张sheet对象化
    sheet1 = excel.sheet_by_name(sheet_names[0])
    # 获取该sheet中的有效行和有效列的数量
    nrows = sheet1.nrows
    ncols = sheet1.ncols
    print("nrows, ncols:", nrows, ncols)
    # 获取该sheet的某一行的所有值，参数最大为nrows-1
    # print("行值", sheet1.row_values(0))
    # 获取该sheet的某一列的所有值，参数最大为ncols-1
    # print("列值", sheet1.col_values(0))
    # 用坐标值获取一个单元格的值，参数最大为(nrows-1,ncols-1)
    # print("单元格值", [sheet1.cell_value(i, col) for col in range(ncols-1)])
    for i in range(nrows):
        question = ''
        for col in range(ncols-1):
            if col <= 3:
                # print(sheet1.cell_value(i,col))
                question += str(sheet1.cell_value(i,col)) + '({})'.format(choices[col])
            else:
                question += str(sheet1.cell_value(i, col)) + '。'+ '正确的选项是：[MASK]'
        # question+='答案是:[gMASK]'
        questions.append(question)
        # print(sheet1.cell_value(i,ncols-1))
        answers.append(sheet1.cell_value(i,ncols-1))

    demonstration = read_5shot(pathfile_dev,choices,n)

    if n>0:
        for i in range(len(questions)):
            query_og = '请参考例子并根据问题在A、B、C、D中选择正确的选项：'+demonstration + questions[i]
            if args.use_10b or args.use_335m:
                if len(query_og) > 1024:
                    demonstration_1 = read_5shot(pathfile_dev,choices,n-1)
                    questions[i] = '请参考例子并根据问题在A、B、C、D中选择正确的选项：'+demonstration_1 + questions[i]
                else:
                    questions[i] = '请参考例子并根据问题在A、B、C、D中选择正确的选项：'+demonstration + questions[i]
            else:
                questions[i] = '请参考例子并根据问题在A、B、C、D中选择正确的选项：'+demonstration + questions[i]

    else:
        for i in range(len(questions)):
            questions[i] = '请根据问题在A、B、C、D中选择正确的选项：' + questions[i]
    # else:
    #     for i in range(len(questions)):
    #         questions[i] = questions[i]

    # 用坐标值获取一个单元格的数量类型，参数最大为(nrows-1,ncols-1)
    # print("单元格的数据类型", [sheet1.cell_type(0, col) for col in range(ncols)])
    print(len(questions),len(answers))
    print("示例:", questions[0],answers[0])
    return questions,answers


if __name__ == "__main__":
    # file = open('total_data_v2.json', 'r',encoding='utf-8')
    # js = file.read()
    # data = json.loads(js)
    args = get_args()
    if args.use_tju:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        path = 'experiments/chatglm-few-shot/'
        dev_path = 'experiments/chatglm-few-shot/'
    else:
        path = 'datammm_untest/'
        dev_path = 'datammm_dev/'
   
    # files = os.listdir(path)
    # files = ['college politics modern history.xlsx']
    files = ['艺术类之绘画.xlsx']
    print(files)
    # n_shot = [0,1,2,3,4,5]
    n_shot = [3,4]

    # model_name = "CPM"

    #6B
    if args.chat:
        if args.use_tju:
            model_path = "models/chatglm-6b"  # You can modify the path for storing the local model
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        else:
            model_path = "models/chatglm-6b"  # You can modify the path for storing the local model
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        directory = "output_mmm_glm6b"
    else:
        if args.use_10b:
            model_path = "/root/autodl-tmp/PLM/glm-10b"  # You can modify the path for storing the local model
            # 10B
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
            model = model.half().cuda()
            directory = "output_mmm_glm10b"
        elif args.belle_7B_2M:
            model_path = "/root/autodl-tmp/PLM/BELLE-7B-2M"  # You can modify the path for storing the local model
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = model.half().cuda()
            directory = "output_mmm_BELLE-7B-2M"
        elif args.belle_7B_02M:
            model_path = "/root/autodl-tmp/PLM/BELLE-7B-0.2M"  # You can modify the path for storing the local model
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = model.half().cuda()
            directory = "output_mmm_BELLE-7B-0.2M"
        elif args.belle_7B_1M:
            model_path = "/root/autodl-tmp/PLM/BELLE-7B-1M"  # You can modify the path for storing the local model
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = model.half().cuda()
            directory = "output_mmm_BELLE-7B-1M"
        elif args.use_335m:
            # 335M
            tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/PLM/glm-large-chinese-335m", trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained("/root/autodl-tmp/PLM/glm-large-chinese-335m", trust_remote_code=True)
            model = model.cuda()
            directory = "output_mmm_glm335m"
    if not os.path.exists(directory):
        os.makedirs(directory)

    if args.test:
        texts = ["你是谁？","你是谁","请介绍一下你自己","请介绍一下你自己。","做一下自我介绍吧","我想知道你是谁","你的名字是什么？","你的名字是什么",
                "你有名字吗？","你会说谎吗？"]
        for text in texts:
            get_text(args,text, model, tokenizer)
    # model_list = "10B/cpm-ant-10b.pt"
    # json_list = "10B/cpm-ant-10b.json"
    else:
        for pathfile in files:
            for n in n_shot:
                corrects = []
                questions, answers = read_excel(path+pathfile,dev_path+pathfile,n)
                correct = 0
                for j,query in enumerate(questions):
                    print(pathfile,n,j)
                    prediction = get_text(args, query, model, tokenizer)
                    print("生成答案:{},正确答案:{}".format(prediction, answers[j]))
                    if str(prediction) == str(answers[j]):
                        correct += 1
                print("正确率:", correct/len(answers))
                corrects.append(str(correct/len(answers)))
                f = open('{}/{}-{}-shot.txt'.format(directory,pathfile.split('.')[0],n),'w')
                f.writelines(' '.join(corrects))
                f.close()


