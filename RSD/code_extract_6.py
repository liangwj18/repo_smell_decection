
import re
import numpy as np
import torch
def extract_classes_and_inheritance(code):
    """
    Extracts all class names and their inheritance relationships from a Java file.

    :param file_path: Path to the Java file.
    :return: A tuple containing a list of class names and a dictionary of inheritance relationships.
    """
    class_pattern = re.compile(r'\bclass\s+(\w+)(?:\s+extends\s+(\w+))?')

    classes = []
    inheritance_map = {}

    lines = code.split('\n')
    for line in lines:
        match = class_pattern.search(line)
        if match:
            class_name = match.group(1)
            parent_class = match.group(2) if match.group(2) else None
            classes.append(class_name)
            inheritance_map[class_name] = parent_class

    return classes, inheritance_map

def build_adjacency_matrix(classes, inheritance_map):
    """
    Builds an adjacency matrix representing class inheritance relationships.

    :param classes: List of class names.
    :param inheritance_map: Dictionary of inheritance relationships.
    :return: Adjacency matrix as a numpy array.
    """
    class_indices = {clss: i for i, clss in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for child, parent in inheritance_map.items():
        if parent and parent in class_indices:
            matrix[class_indices[parent], class_indices[child]] = 1

    return matrix
    

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


# -------------binary_classification------------------------------------------------#
binary_folder = "./baseline1_dataset/baseline_1/binary_classification"
output_folder = "./RSD/RSD_dataset/binary_classification"
backbone_folder = "./RSD/RSD_dataset/binary_classication_backbone"
summary_folder = "./RSD/RSD_dataset/binary_classication_summary"
generation_folder = "./RSD/RSD_dataset/binary_classication_generation"
goodedge_folder = "./RSD/RSD_pretreatment/"

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
   
    "Feature Envy":1,
    "Rebellious Hierarchy":1,
  
    }
    
    
    for smell_name in smellsss:
        data = []
        data_path = os.path.join(generation_folder, smell_name+"_"+benchmark+".jsonl")
        output_path = os.path.join(goodedge_folder, f"binary_classification_{smell_name}_{benchmark}")
        if os.path.exists(output_path) == False:
            os.makedirs(output_path)
        tasks = read_task_in_jsonl(data_path)
        for task in tasks:
            classes, inheritance_map = extract_classes_and_inheritance(task)
            matrix = build_adjacency_matrix(classes, inheritance_map)
            for i in range(len(matrix[0])):
                matrix[i][i] = 1
            
            data.append(torch.tensor(matrix))
        torch.save(data, os.path.join(output_path,"good_edge.pth"))
       
            
if __name__ == "__main__":
    main()
  