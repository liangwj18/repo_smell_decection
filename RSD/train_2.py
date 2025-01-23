import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(parent_dir)
from utils_code.utils_0 import read_task_in_jsonl, output_jsonl, useful_design_smell, split_smell, merge_smell
from tqdm import tqdm
from location_tool.repo_locate_tool_builder import repo_locate_tool_build
import subprocess
import time

# -------------binary_classification------------------------------------------------#
binary_folder = "./RSD/RSD_dataset/binary_classification"
output_folder = "./RSD/RSD_dataset/binary_classification_result"
models_path = "./RSD/model/distilbert-base-uncased"
model_name = models_path.split("/")[-1]


def get_free_gpus(min_free=2,timestamps=None):

    try:

        gpustat_output = subprocess.check_output("gpustat --json", shell=True).decode("utf-8")
        import json
        gpu_status = json.loads(gpustat_output)
        # print(gpu_status)
    
        free_gpus = [str(i) for i, gpu in enumerate(gpu_status['gpus']) if True or (gpu['memory.used'] < 200 and int(time.time())-timestamps[i] > 20 )]
        

        if len(free_gpus) >= min_free:
            return free_gpus[:min_free]
        else:
            return []
    except Exception as e:
        print(f"Error in checking GPUs: {e}")
        return []
    
from multiprocessing import Process
import time

def start_test(gpus, smell_type, data_path, output_folder, benchmark, exp_name):

    gpu_str = ",".join([str(x) for x in gpus])
    if os.path.exists(output_folder) == False:
        os.makedirs(output_folder)
    port = str(12365 + int(gpus[0]))
    test_command = f"""export CUDA_VISIBLE_DEVICES={gpu_str}\npython -m RSD.train_detail_2_0 --smell_type "{smell_type}" --task_file_path "{data_path}" --model_name_or_path {models_path} --output_folder {output_folder} --port {port} --benchmark {benchmark} --exp_name {exp_name}"""
    
   
    try:
      
        print(test_command)
        subprocess.run(test_command, shell = True)
        # stdout, stderr = process.communicate()
        # print("Output:", stdout.decode())
    except Exception as e:
        print(f"Error in starting tmux session: {e}")
    # return process

# 主程序

def binary_train():
    benchmark = 'train'
    exp_name = "full"
    
    output_path = os.path.join(output_folder, model_name, benchmark)
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
    processes = []
    timestamps = [0 for x in range(8)]
    each_train_min_gpus = 1
    # tmp_design_smell = useful_design_smell
    tmp_design_smell = {
        "Cyclic Hierarchy",
        "Broken Hierarchy",
        "Cyclically-dependent Modularization",
        "Deep Hierarchy",
        "Wide Hierarchy",
        "Feature Envy",
        "Multipath Hierarchy",
        "Rebellious Hierarchy",
    }
 
    try:
        for smell_type in tmp_design_smell:
            data_path = os.path.join(binary_folder, smell_type+f"_{benchmark}.jsonl")
            free_gpus = None
            while free_gpus is None or len(free_gpus) < each_train_min_gpus:
                if free_gpus is not None:
                    time.sleep(30)
                free_gpus = get_free_gpus(min_free=each_train_min_gpus,timestamps = timestamps)
                print(free_gpus)
            
            current_timestamp = int(time.time())
            for x in free_gpus:
                timestamps[int(x)] = current_timestamp
            
            p = Process(target=start_test, args=(free_gpus, smell_type, data_path, output_path, benchmark, exp_name))
            processes.append(p)
            p.start()  # 启动进程
            time.sleep(15)
            
    except Exception as e:
        print(e)
        for p in processes:
            p.kill()
    for p in processes:
        p.join()
  

def main():
    binary_train()

if __name__ == "__main__":
    main()