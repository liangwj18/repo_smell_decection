

from location_tool.repo_locate_tool_builder import repo_locate_tool_build
from location_tool.repo_locate_tool import extract_class_code, extract_function_code
from utils_code.utils_0 import useful_design_smell, read_task_in_jsonl, output_jsonl, smell_description_dic

from tqdm import tqdm
import os
import copy
import json
import subprocess
from multiprocessing import Process
import time

repo_base_dir = "./design_smell_4k_GITHUB-REPO"
# -------------binary_classification------------------------------------------------#
binary_folder = "./baseline1_dataset/baseline_1/binary_classification"
output_folder = "./RSD/RSD_dataset/binary_classification"
backbone_folder = "./RSD/RSD_dataset/binary_classication_backbone"
summary_folder = "./RSD/RSD_dataset/binary_classication_summary"
generation_folder = "./RSD/RSD_dataset/binary_classication_generation"

from openai import OpenAI
client = OpenAI(
    api_key="<YOUR API KEY>",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
def get_prompt_dataset(st):
   
 
    
    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'user', 'content': st + "\nPlease generate a code framework based on the above description, where each class must and can only retain its definition and inheritance, and use brief comments to describe the role of the class."},
            ],
    )
    refine_prompt = json.loads(completion.model_dump_json())['choices'][0]['message']['content']
    return refine_prompt

def generation(smell_name, tasks, generations,  benchmark):
    data_path = os.path.join(backbone_folder, smell_name+"_"+benchmark+".jsonl")
    tasks[smell_name] = read_task_in_jsonl(data_path)
    print(os.path.join(generation_folder, smell_name+"_"+benchmark+".jsonl"))
    generations[smell_name] = read_task_in_jsonl(os.path.join(generation_folder, smell_name+"_"+benchmark+".jsonl"))
    while len(generations[smell_name]) < len(tasks[smell_name]):
        summary_path = os.path.join(summary_folder, smell_name+"_"+benchmark+".jsonl")
        summarys = read_task_in_jsonl(summary_path)
        for i in tqdm(range(len(generations[smell_name]), len(summarys))):
            generations[smell_name].append(get_prompt_dataset(summarys[i]))
            if len(generations[smell_name]) == 5:
                break
        print(smell_name, len(generations[smell_name]), "/", len(tasks[smell_name]))
        output_jsonl(generations[smell_name], os.path.join(generation_folder, smell_name+"_"+benchmark+"2.jsonl"))
        time.sleep(60)

def main():
    benchmark = "train"
    finish_smell = {}
    smellsss = {
    "Deep Hierarchy":1,
    "Wide Hierarchy":1,
    "Broken Hierarchy":1,
    "Cyclic Hierarchy":1,
    "Cyclically-dependent Modularization":1,
    "Multipath Hierarchy":1,
    # "Wide Hierarchy":1,
    # "Deep Hierarchy":1,
    "Feature Envy":1,
    "Rebellious Hierarchy":1,
    # "Duplicate Abstraction":1
    }
    generations = {
    "Deep Hierarchy":[],
    "Wide Hierarchy":[],
    "Broken Hierarchy":[],
    "Cyclic Hierarchy":[],
    "Cyclically-dependent Modularization":[],
    "Multipath Hierarchy":[],
    # "Wide Hierarchy":[],
    # "Deep Hierarchy":[],
    "Feature Envy":[],
    "Rebellious Hierarchy":[],
    }
    processes = []
    
    tasks = {
        
    }
    if os.path.exists(generation_folder) == False:
        os.makedirs(generation_folder)
    try:
        for smell_name in smellsss:
            p = Process(target=generation, args=(smell_name, tasks, generations, benchmark))
            processes.append(p)
            p.start()  # 启动进程            
    except Exception as e:
        print(e)
        for p in processes:
            p.kill()
    for p in processes:
        p.join()
   
       
            
if __name__ == "__main__":
    main()
 