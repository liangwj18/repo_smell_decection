from location_tool.repo_locate_tool_builder import repo_locate_tool_build
from location_tool.repo_locate_tool import extract_class_code, extract_function_code
from utils_code.utils_0 import useful_design_smell, read_task_in_jsonl, output_jsonl, smell_description_dic

from tqdm import tqdm
import os
import copy

repo_base_dir = "./design_smell_4k_GITHUB-REPO"
# -------------binary_classification------------------------------------------------#
binary_folder = "./baseline1_dataset/baseline_1/binary_classification"
output_folder = "./RSD/RSD_dataset/binary_classification"

import re

# 图的容器，存储所有的顶点和边
class GATCLASS:
    def __init__(self):
        self.package = package
        self.class_name = class_name
        self.class_code = ""
        self.methods = []

  
# 定义正则表达式，用于提取Package, Class, Method, Variable
PACKAGE_REGEX = re.compile(r'package\s+([\w\.]+);')
CLASS_REGEX = re.compile(r'class\s+(\w+)')

VARIABLE_REGEX = re.compile(r'(\w+)\s+(\w+)\s*(;|=)')
INHERITANCE_REGEX = re.compile(r'class\s+(\w+)\s+extends\s+(\w+)')

def find_inherit_forest_cls(center_package, center_class, location_tool):
    visited = {}
    queue = [[center_package, center_class]]
    edges = [[0 for x in range(1500)] for y in range(1500)]
    head = 0
    tail = 1
    while head < tail and head < 1000 and tail < 1000:
        u_package, u_class = queue[head]
        edges.append([])
        head += 1
        key = u_package + "!" +u_class
        visited[key] = 1
        if key in location_tool.fathers:
            for father in location_tool.fathers[key]:
                fa_package, fa_class = father.split("!")
                if location_tool.locate_file(fa_package, fa_class)[0] is None:
                    continue
                if father not in visited:
                    edges[tail][head - 1] = 1
                    #TODO 有向边
                    visited[father] = 1
                    tail += 1
                    queue.append([fa_package,fa_class])
        if key in location_tool.sons:
            for son in location_tool.sons[key]:
                son_package, son_class = son.split("!")
                if location_tool.locate_file(son_package, son_class)[0] is None:
                    continue
                if son not in visited:
                    visited[son] = 1
                    edges[head - 1][tail] = 1
                    tail += 1
                    queue.append([son_package,son_class])
    cut_edges = []
    for j in range(tail):
        cut_edges.append(edges[j][:tail])
    return [queue, cut_edges]

def find_use_Question_cls(center_package, center_class, location_tool):
    result_cls = [[center_package, center_class]]
    for i in range(len(location_tool.all_repo_files)):
        package_name, class_name, path = location_tool.all_repo_files[i]
        if package_name == center_package and class_name == center_class:
            continue
        if location_tool.check_use(package_name, class_name, center_package, center_class):
            result_cls.append([package_name, class_name])
    return result_cls

# 主函数：处理所有文件，构建图
def build_code_graph(center_package, center_class, location_tool):

    inherit_forest = find_inherit_forest_cls(center_package, center_class, location_tool) 
    #vertexlist edge_list
    #vertex is [package, class]
    #edge is [V * V]
    
    use_cls = find_use_Question_cls(center_package, center_class, location_tool)
    #use_cls [package, class] #0 is question
    use_cls_members = {}
    
    merged_list = copy.deepcopy(use_cls)
    for a,b in inherit_forest[0]:
        merged_list.append([a,b])
        
    for package_name, cls_name in merged_list:
        # print(package_name,cls_name)
        key = package_name +"!"+cls_name
        if key not in use_cls_members:
            use_cls_members[key] = location_tool.get_members(package_name, cls_name)
    #use_cls_members:
    #["method_name",method_code,method_return_line]" #0 is question
    use_members_statment = [None]
    for package_name, cls_name in use_cls[1:]:
        members = [
            use_cls_members[center_package+"!"+center_class],
            use_cls_members[package_name+"!"+cls_name]
        ]
        use_members_statment.append(location_tool.find_two_cls_use_members(members))
    
    #use_members_statment:
    #0 is question 
    #map[2][LEN_OF_METHOD] 0 is q'class code ,other's functions and 1 is other
    #map[2][LEN_OF_METHOD] =  use_line
    return {"inherit_forest":inherit_forest, "use_cls":use_cls, "cls_members_dict":use_cls_members, "use_members_statment":use_members_statment}


def build_graph(task, tmp_location_tool):
    metadata = task["metadata"]
    repo_name = metadata["repo"]
    if tmp_location_tool is None or tmp_location_tool.repo_name != repo_name:
        tmp_location_tool = repo_locate_tool_build('java',os.path.join(repo_base_dir, repo_name))
    data_dict = build_code_graph(metadata['package'], metadata['class'], tmp_location_tool)
    return data_dict
    # return {}

def multi_label():
    benchmark = "train_only_metadata"
    
    data = []
    data_path = os.path.join(multi_label_folder,benchmark+".jsonl")
    output_path = os.path.join(multi_label_output_folder, benchmark+".jsonl")
    tasks = read_task_in_jsonl(data_path)
    tmp_location_tool = None
    tasks.sort(key = lambda x:x['metadata']['repo'])
    for i in tqdm(range(len(tasks))):
        task = tasks[i]
        input_data = build_graph(task, tmp_location_tool)
        output_data = task['label']
        data.append({"data":input_data,"label":output_data})
        if i % 100 == 0:
            
            if os.path.exists(multi_label_output_folder) == False:
                os.makedirs(multi_label_output_folder)
            output_jsonl(data,output_path)
    output_jsonl(data,output_path)
def main():
    benchmark = "train"
    smellsss = {
    "Deep Hierarchy":1,
    "Wide Hierarchy":1
     "Broken Hierarchy":1,
    "Cyclic Hierarchy":1,
    "Cyclically-dependent Modularization":1,
    "Multipath Hierarchy":1,
    # "Wide Hierarchy":1,
    # "Deep Hierarchy":1,
    "Feature Envy":1,
    "Rebellious Hierarchy":1,
    "Duplicate Abstraction":1
    }
    for smell_name in smellsss:
        data = []
        data_path = os.path.join(binary_folder, smell_name+"_"+benchmark+".jsonl")
        tasks = read_task_in_jsonl(data_path)
        tmp_location_tool = None
        tasks.sort(key = lambda x:x['metadata']['repo'])
        for i in tqdm(range(len(tasks))):
            task = tasks[i]
            input_data = build_graph(task, tmp_location_tool)
            output_data = task['label']
            data.append({"data":input_data,"label":output_data})
            if i % 100 == 0:
                output_path = os.path.join(output_folder, smell_name +"_"+benchmark+".jsonl")
                if os.path.exists(output_folder) == False:
                    os.makedirs(output_folder)
                output_jsonl(data,output_path)
        output_path = os.path.join(output_folder, smell_name +"_"+benchmark+".jsonl")
        output_jsonl(data,output_path)

if __name__ == "__main__":
    main()
