import os
class repo_locate_tool():
    
    def collect_all_repo_files(self):
        pass
    def check_package_class(self, file_content, package_name, class_name):
        pass
    def locate_class_inner_content_by_line(self, file_content, class_name, line_num):
        pass
    def locate_class_inner_content_by_length(self, file_content, class_name, length):
        pass
    def read_package_name(self, file_content):
        pass
    def get_father(self, class_name, file_content, path):
        pass
    def __init__(self,repo_local_path, build_connection_graph):
        # print(repo_local_path)
        self.repo_name = repo_local_path.split("/")[-1]
        self.repo_local_path = repo_local_path
        self.build_connection_graph = build_connection_graph
        # print(self.repo_local_path)
        self.sons = {}
        self.fathers = {}
        self.collect_all_repo_files()
        if self.build_connection_graph:
            self.build_connection_Graph()
    
    def build_connection_Graph(self):
        for i in range(len(self.all_repo_files)):
            if len(self.all_repo_files[i])!=3:
                print(self.all_repo_files[i])
            package_name, class_name, path = self.all_repo_files[i]
         
            content = read_file(path)
            fas = self.get_father(class_name, content, path)
            for fa_package, fa_class in fas:
                if fa_package == "(default package)" and fa_class == "WebViewClientDelegate":
                    print(content)
                    print(self.repo_local_path)
                    assert 1 == 2
                if fa_package == None or fa_class == None:
                    continue
                insert(self.fathers, package_name + "!" + class_name, fa_package + "!" + fa_class)
                insert(self.sons,  fa_package + "!" + fa_class, package_name + "!" + class_name)
    
    def locate_start_end_line(self, package_name, class_name):
        file_content, file_path = self.locate_file(package_name, class_name)
        class_content, class_start_line, class_end_line = self.locate_class_inner_content_by_line(file_content, class_name, 500)
        fpath_tuple = file_path.split("/")
        fpath_start_idx = -1
        for i in range(len(fpath_tuple)):
            if fpath_tuple[i] == self.repo_name:
                fpath_start_idx = i 
                break
        fpath_tuple = fpath_tuple[fpath_start_idx:]
        return fpath_tuple, class_start_line, class_end_line, len(class_content)
    
    def locate_file(self, package_name, class_name):
        index = binary_search(self.all_repo_files, class_name)
        # print("package_name, class_name", package_name, class_name)
        # for i in range(len(self.all_repo_files)):
        #     print(self.all_repo_files[i][0], self.all_repo_files[i][1])

        if index == -1:
            return [None, None]
        lef_index = index
        while lef_index>0 and self.all_repo_files[lef_index-1][1] == class_name:
            lef_index -= 1
        for i in range(lef_index, len(self.all_repo_files)):
            # print(self.all_repo_files[i][0], self.all_repo_files[i][1])
            if self.all_repo_files[i][1] != class_name:
                return [None, None] # private class
            file_content = read_file(self.all_repo_files[i][2])
            if package_name == self.all_repo_files[i][0]:
                return [file_content, self.all_repo_files[i][2]]
            # print(self.all_repo_files[i][1], class_name, package_name)
        return [None, None] #external import
        # print(class_name, package_name, self.all_repo_files)
        # assert 1 == 2
    def locate_same_package_class(self, package_name, class_name):
        file_path = self.locate_file(package_name, class_name)[1]
        same_folder = "/".join(file_path.split("/")[:-1])
        result = []
        for other_file in os.listdir(same_folder):
            if other_file.endswith(self.suffix) == False:
                continue
            other_class_name = other_file.split(self.suffix)[0]
            other_file_path = os.path.join(same_folder, other_file)
            if other_file_path == file_path:
                continue
            result.append([other_file_path, [package_name, other_class_name]])
        return result
    
    def baseline_1_input_binary(self, package_name, class_name, MAX_LENGTH, MAX_N, smell_type, reuse_prompt = None):
        pass
    def baseline_1_input_multilabel(self, package_name, class_name, MAX_LENGTH, MAX_N, smell_list):
        pass
    def baseline_2_input_multilabel(self, package_name, class_name, contexts, MAX_LENGTH, MAX_N, SIM_SCORE_THRESHOLD, smell_list):
        pass
    def locate_all_inheritance_name(self, package_name, class_name):
        pass



def insert(dic, K, V):
    if K not in dic:
        dic[K] = {}
    if V not in dic[K]:
        dic[K][V] = 1

def Recursive_Traversal_Files(path, endswith = None):
    file_names = [] 
    items = os.listdir(path)
    for item in items:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            file_names.extend(Recursive_Traversal_Files(item_path, endswith))
        elif endswith is None or item_path.endswith(endswith):
            file_names.append(item_path)
    return file_names

# def deduplicate_pack_and_class(pack_and_classes):
#     merge_list = [x[0]+"!"+x[1]+"!"+x[2] for x in pack_and_classes]
#     merge_list.sort()
#     result_list = []
    
#     for x in merge_lsit:
#     result_list = [x.split("!") for x in merge_list]
#     return result_list

def deduplicate_str(str_list):
    s = set()
    new_str_list = []
    for x in str_list:
        if x not in s:
            new_str_list.append(x)
            s.add(x)
    return new_str_list

def binary_search(listA, value, start = -1, end = -1):
    lef = start if start > -1 else 0
    rig = end if end > -1 else len(listA)-1
    while lef <= rig:
        mid = (lef + rig) // 2
        if listA[mid][1] == value:
            return mid
        elif listA[mid][1] > value:
            rig = mid - 1
        else:
            lef = mid + 1
    return -1

def read_file(path):
    try:
        return "".join(open(path,'r').readlines())
    except:
        return None

def find_content_in_lines(content, lines):
    for i in range(len(lines)):
        if lines[i].find(content)!=-1:
            return i
import re

def is_class_a_calling_class_b(class_a_content, class_b_name):
    """
    Determine if Class A calls Class B's members.

    Parameters:
        class_a_content (str): The content of Class A.
        class_b_name (str): The name of Class B.

    Returns:
        bool: True if Class A calls Class B's members, otherwise False.
    """
    pattern = rf"\b{class_b_name}\b\.\w+"
    return bool(re.search(pattern, class_a_content))
def extract_class_code(java_source_text: str, class_name: str) -> str:
  
    search_pattern = f"class {class_name}"


    start_idx = java_source_text.find(search_pattern)
    if start_idx == -1:
   
        return ""

    brace_idx = java_source_text.find("{", start_idx)
    if brace_idx == -1:
     
        return ""

    bracket_count = 0
    end_idx = -1

   
    for i in range(brace_idx, len(java_source_text)):
        char = java_source_text[i]
        if char == '{':
            bracket_count += 1
        elif char == '}':
            bracket_count -= 1
        
      
        if bracket_count == 0:
            end_idx = i  
            break

    if end_idx == -1:
        return ""

 
    class_definition = java_source_text[start_idx : end_idx + 1]
    return class_definition
    
def extract_function_code(java_code, function_name):

    function_pattern = re.compile(r'(\w[\w\<\>\[\]]*\s+)+({}\s*\(.*?\)\s*)\{{(.*?)\}}'.format(re.escape(function_name)), re.DOTALL)

    match = function_pattern.search(java_code)
    
    if match:

        return match.group(0)
    else:
        return None
    
def extract_member_variables(class_content):
    """
    Extract member variable declarations from a class content.

    Parameters:
        class_content (str): The content of the class.

    Returns:
        list: A list of member variable declarations.
    """
    pattern = r"(?:private|protected|public)\s+[^;=]+;"
    res = re.findall(pattern, class_content)
    new_res = []
    for x in res:
        if x.find("\n")==-1:
            new_res.append(x)
    return "\n".join(new_res)