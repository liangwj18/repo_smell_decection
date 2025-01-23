import re

def extract_class_and_methods(java_content):
    """
    Extract class definitions and method definitions from a Java file.
    :param java_file_path: Path to the input Java file.
    :param output_file_path: Path to save the output with class and method definitions.
    """
    lines = java_content.split("\n")
    
    # Regular expressions to match class and method definitions
    class_pattern = re.compile(r'\b(class|interface|enum)\s+\w+')
    method_pattern = re.compile(r'\b(public|protected|private|static|final|abstract|synchronized)\s+.*\b\w+\s*\(.*\)\s*{?')
    
    extracted_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        if class_pattern.search(stripped_line) or method_pattern.search(stripped_line):
            extracted_lines.append(line)
    
    # Save the extracted lines to a new file
    return "\n".join(extracted_lines)

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
backbone_folder = "./RSD/RSD_dataset/binary_classication_backbone"

def get_backbone(task,taskV,tmp_location_tool):
    metadata = task["metadata"]
    repo_name = metadata["repo"]
    if tmp_location_tool is None or tmp_location_tool.repo_name != repo_name:
        tmp_location_tool = repo_locate_tool_build('java',os.path.join(repo_base_dir, repo_name), False)
    V,E = taskV["data"]["inherit_forest"]
    V = V[:80]
    backbones = []
    for pack, clss in V:
        content = tmp_location_tool.locate_file(pack, clss)[0]
        backbone = extract_class_and_methods(content)
        backbones.append(backbone)
    return backbones
def main():
    benchmark = "train"
    smellsss = {
    "Deep Hierarchy":1,
    "Wide Hierarchy":1,
    "Broken Hierarchy":1,
    "Cyclic Hierarchy":1,
    "Cyclically-dependent Modularization":1,
    "Multipath Hierarchy":1,

    "Feature Envy":1,
    "Rebellious Hierarchy":1,
    "Duplicate Abstraction":1
    }
    if os.path.exists(backbone_folder) == False:
        os.makedirs(backbone_folder)
    for smell_name in smellsss:
        data = []
        data_path = os.path.join(binary_folder, smell_name+"_"+benchmark+".jsonl")
        tasks = read_task_in_jsonl(data_path)
        task_with_vertexs = read_task_in_jsonl(os.path.join(output_folder,smell_name +"_"+benchmark+".jsonl"))
      
        tmp_location_tool = None
        tasks.sort(key = lambda x:x['metadata']['repo'])
        for i in tqdm(range(len(task_with_vertexs))):
            task = tasks[i]
            backbone = get_backbone(task,task_with_vertexs[i], tmp_location_tool)
            
            data.append({"backbone":backbone})
        
        output_path = os.path.join(backbone_folder, smell_name +"_"+benchmark+".jsonl")
        output_jsonl(data,output_path)

if __name__ == "__main__":
    main()
