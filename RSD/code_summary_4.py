

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

from openai import OpenAI

def get_prompt_dataset(backbones, output_path):
   
    t = []
    client = OpenAI(
        api_key="<YOUR API>",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    for i in tqdm(range(len(backbones))):
        backbone = backbones[i]
        st = "\n".join(backbone)
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {'role': 'user', 'content': 'Please summarize the overall functionality of the code skeleton below, without specific class names:\n' + st},
                ],
        )
        refine_prompt = json.loads(completion.model_dump_json())['choices'][0]['message']['content']
        t.append(refine_prompt)
        if i % 20 == 0 or i + 1 == len(backbones):
            output_jsonl(t, output_path)

def main():
    benchmark = "test"
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
    processes = []
   
    try:
        if os.path.exists(summary_folder) == False:
            os.makedirs(summary_folder)
        for smell_name in smellsss:
            data = []
            data_path = os.path.join(backbone_folder, smell_name+"_"+benchmark+".jsonl")
            output_path = os.path.join(summary_folder, smell_name+"_"+benchmark+".jsonl")
            tasks = read_task_in_jsonl(data_path)
            p = Process(target=get_prompt_dataset, args=(tasks, output_path))
            processes.append(p)
            p.start()  # 启动进程
           
            time.sleep(5)
            
    except Exception as e:
        print(e)
        for p in processes:
            p.kill()
    for p in processes:
        p.join()
   
       
            
if __name__ == "__main__":
    main()
