from .repo_locate_tool import repo_locate_tool, Recursive_Traversal_Files, find_content_in_lines, read_file, extract_class_code, is_class_a_calling_class_b, extract_function_code, extract_member_variables

import os
import re
class repo_locate_tool_java(repo_locate_tool):
    def __init__(self, repo_local_path, build_connection_graph):
        super().__init__(repo_local_path, build_connection_graph)
        self.suffix = ".java"
    def read_package_name(self, content):
        package_pattern = r'package\s+([^\s;]+);'

        package_match = re.search(package_pattern, content)
        
        if package_match:
            return package_match.group(1)
        else:
            return "(default package)"
    # def get_father(self, class_name, file_content):
    #        # "public class PG extends Student "
    #     search_pattern = class_name + " extends "
      
    #     extend_idx = file_content.find(search_pattern) + len(search_pattern)
    #     if extend_idx == -1:
    #         search_pattern = class_name + " implements "
    #         extend_idx = file_content.find(search_pattern) + len(search_pattern)
    #     if extend_idx == -1:
    #         return None, None
    #     else:
    #         end_idx = extend_idx
    #         while file_content[end_idx] != ' ' and file_content[end_idx]!='<' and file_content[end_idx]!='{' and file_content[end_idx]!='\n':
    #             end_idx += 1
    #         father_class = file_content[extend_idx:end_idx]
    #         # if len(father_class) > 1 and father_class[-1] == ")":
    #         print(file_content)
    #         print("-"*20)
    #         print("class_name",class_name)
    #         print(search_pattern)
    #         print("father_class",father_class)
    #         print(file_content[extend_idx:end_idx+50])
    #         print(file_content[:end_idx+50])
    #         print("-"*20)
            
    #         pattern = r'import\s+(.*?)'+father_class
    #         father_package = re.findall(pattern, file_content)
    #         if len(father_package) == 0:
    #             father_package = "(default package)"
    #         else:
    #             father_package = father_package[0][:-1]
    #         assert 1 == 2
    #         return father_package, father_class
    def get_father(self, class_name, file_content, path):
        
        def check_same_package(file_path, cls_name, package_name):
            folder_path = "/".join(file_path.split("/")[:-1])
            for name in os.listdir(folder_path):
                if name == class_name + ".java":
                    return package_name
            return "(default package)"
        
 
        class_pattern = rf'class\s+{class_name}\s+extends\s+([^\s]+)\s+implements\s+([^\s]+)'
        extends_pattern = rf'class\s+{class_name}\s+extends\s+([^\s]+)'
        implements_pattern = rf'class\s+{class_name}\s+implements\s+([^\s]+)'

        import_pattern = r'import\s+([^\s;]+);'
        java_code = file_content
        imports = re.findall(import_pattern, java_code)
        
        class_info = {}

        class_match = re.search(class_pattern, java_code)
        extends_match = re.search(extends_pattern, java_code)
        implements_match = re.search(implements_pattern, java_code)
    
        if class_match:
            class_info['extends'] = class_match.group(1)
            class_info['implements'] = class_match.group(2)
        elif extends_match:
            class_info['extends'] = extends_match.group(1)
        elif implements_match:
            class_info['implements'] = implements_match.group(1)
        result = []
        package_name = self.read_package_name(file_content)

        if 'extends' in class_info:
            extends_import = next((imp for imp in imports if class_info['extends'] in imp), '(default package).')
            extends_package = ".".join(extends_import.split(".")[:-1])
            if extends_package == "(default package)":
                extends_package = check_same_package(path, class_info['extends'], package_name)
            result.append([extends_package, class_info['extends']])
        if 'implements' in class_info:
            implements_import = next((imp for imp in imports if class_info['implements'] in imp), '(default package).')
            implements_package = ".".join(implements_import.split(".")[:-1])
            if implements_package == "(default package)":
                implements_package = check_same_package(path, class_info['implements'], package_name)
            result.append([implements_package, class_info['implements']])
        return result


    def collect_all_repo_files(self):
        self.all_repo_files = Recursive_Traversal_Files(self.repo_local_path, '.java')
        self.all_repo_files = [[x.split("/")[-1].split(".java")[0], x] for x in self.all_repo_files]
        self.new_all_repo_files = []
        
        for i in range(len(self.all_repo_files)):
            class_name, path = self.all_repo_files[i]
            content = read_file(path)
            if content is None:
                continue
            package_name = self.read_package_name(content)
            self.new_all_repo_files.append([package_name, class_name, path])
        self.all_repo_files = self.new_all_repo_files
        self.all_repo_files.sort(key = lambda x: x[1])

    def check_package_class(self, file_content, package_name, class_name):
        if package_name != "(default package)" and file_content.find("package "+package_name) == -1:
            return False
        if file_content.find("class "+class_name) != -1:
            return True
        else:
            return False
    def locate_class_inner_content_by_line(self, file_content, class_name, line_num):
        lines = file_content.split("\n")
        class_definition_start_line = find_content_in_lines("class "+class_name, lines)
        try:
            return ["\n".join(lines[class_definition_start_line:class_definition_start_line + line_num]), class_definition_start_line, class_definition_start_line + line_num]
        except:
            print(file_content)
            assert 1 == 2
    
    def locate_class_inner_content_by_length(self, file_content, class_name, length):
   
        class_definition_start_idx = file_content.find("class "+class_name,)
        
        return file_content[class_definition_start_idx:class_definition_start_idx + length]
    
 
    def locate_all_inheritance_name(self, package_name, class_name):
        pass
    
    def baseline_1_code(self, package_name, class_name, MAX_LENGTH, MAX_N):
        def clean_external_classes(classes):
            new_classes = []
            for pack_name, cls_name, type_idx in classes:
                # if self.locate_file(pack_name, cls_name) is None:
                #     print(self.locate_file(pack_name, cls_name))
                #     assert 1 == 2
                if self.locate_file(pack_name, cls_name)[0] is None:
                    continue
                new_classes.append([pack_name, cls_name,
                                    type_idx])
            return new_classes   
        class_itself_rate = 0.3
        class_file_content = self.locate_file(package_name, class_name)[0]
        prompt = "Tip: The content in the <JAVA> </JAVA> tag is the JAVA code content that needs to be referenced.\n"
        prompt += "Please read the following JAVA code:\n" +"<JAVA> "+self.locate_class_inner_content_by_length(class_file_content, class_name, int(MAX_LENGTH * class_itself_rate)) +" </JAVA>\n"
        
        key = package_name + "!" + class_name
        classes = []
        
        in_class_dic = {}
        if key in self.fathers:
            for father in self.fathers[key]:
                fa_package, fa_class = father.split("!")
                if father not in in_class_dic:
                    in_class_dic[father] = 1
                    classes.append([fa_package,fa_class,0])
        # print(self.locate_file(package_name, class_name)[1])
        if key in self.sons:
            for son in self.sons[key]:
                son_package, son_class = son.split("!")
                if son not in in_class_dic:
                    in_class_dic[son] = 1
                    classes.append([son_package,son_class,1])
                
        # print(classes)
        for bro in self.locate_same_package_class(package_name, class_name):
            bro_package, bro_class = bro[1]
            bro_s = bro_package + "!" + bro_class
            if bro_s not in in_class_dic:
                in_class_dic[bro_s] = 1
                classes.append([bro_package,bro_class,2])
         
        # print(classes)
        # classes = deduplicate_pack_and_class(classes)
        classes = clean_external_classes(classes)
        classes = classes[:MAX_N]
        N = len(classes)
        
        Type_Prompts = [
            "The code snippets directly imported by Class {}(the fathers of Class {}) are as follows:\n".format(class_name, class_name),
            "The code snippets directly referencing Class {}(the sons of Class {}) are as follows:\n".format(class_name, class_name),
            "The code snippets belonging to the same package as class {} are as follows:\n".format(class_name)
        ]
       
        last_type = -1
        for i in range(N):
            class_file_code = self.locate_file(classes[i][0], classes[i][1])[0]
            if class_file_code is None: # 引用了外部
                # print(classes[i], self.repo_local_path)
                # for x in self.all_repo_files:
                #     print(x)
                assert 1 == 2
            if classes[i][2] != last_type:
                last_type = classes[i][2]
                prompt += Type_Prompts[last_type]
            class_code = self.locate_class_inner_content_by_length(class_file_code, classes[i][1], int(MAX_LENGTH * (1 - class_itself_rate)) // N)
            prompt += "<JAVA> " + class_code + " </JAVA>\n"
        return prompt, N
    
    def baseline_1_input_binary(self, package_name, class_name, MAX_LENGTH, MAX_N, smell_type, reuse_prompt = None):
        if self.build_connection_graph == False:
            assert 1 == 2
            
        if smell_type is not None:
            Question = "Please analyze if there is a code smell {} present in above code, answer in YES or NO.\n".format(smell_type)
        else:
            assert 1 == 2
            # Question = "Please analyze what kind of code smell exists in the above code."
        
        if reuse_prompt is not None:
            return reuse_prompt + Question, 0, reuse_prompt
        # class_file_itself
       
        prompt, N = self.baseline_1_code(package_name, class_name, MAX_LENGTH, MAX_N)
        reuse_prompt = prompt
        prompt += Question
        
        return prompt, N, reuse_prompt
    def baseline_1_input_multilabel(self, package_name, class_name, MAX_LENGTH, MAX_N, smell_list):
        if self.build_connection_graph == False:
            assert 1 == 2
            
        if smell_list is not None:
            Question = "Here is a list of code smells you need to consider: " + ", ".join(smell_list) + "\n" +"For each code smell, please analyze separately whether the above code contains it, answer in YES or NO.\n"
        else:
            assert 1 == 2
            # Question = "Please analyze what kind of code smell exists in the above code."
        
        prompt, N = self.baseline_1_code(package_name, class_name, MAX_LENGTH, MAX_N)
        reuse_prompt = prompt
        prompt += Question
        
        return prompt, N
    
    def baseline_2_input_multilabel(self, package_name, class_name, contexts, MAX_LENGTH, MAX_N, SIM_SCORE_THRESHOLD, smell_list):
        Question = "Here is a list of code smells you need to consider: " + ", ".join(smell_list) + "\n" +"For each code smell, please analyze separately whether the above code contains it, answer in YES or NO.\n"
        class_file_content = self.locate_file(package_name, class_name)[0]
        prompt = "Tip: The content in the <JAVA> </JAVA> tag is the JAVA code content that needs to be referenced.\n"
        class_itself_rate = 0.4
        prompt += "Please read the following JAVA code:\n" +"<JAVA> "+self.locate_class_inner_content_by_length(class_file_content, class_name, int(MAX_LENGTH * class_itself_rate)) +" </JAVA>\n"
        use_contexts = []
        for k in range(0,min(len(contexts), MAX_N)):
       
            if contexts[k]['sim_score'] < SIM_SCORE_THRESHOLD:
                break
            use_contexts.append(contexts[k]["context"])
        N = len(use_contexts)
        use_contexts = [x[0:int(MAX_LENGTH * (1 - class_itself_rate)) // N] for x in use_contexts]
        prompt += "The TOP-K code snippets similar to CLASS {} obtained through semantic similarity retrieval are as follows:\n".format(class_name)
        for i in range(N):
            prompt += "<JAVA> " + use_contexts[i] + " </JAVA>\n"
        prompt += Question
        return prompt, N
    
    def check_use(self, package_name, class_name, center_package, center_class):
        A_content = self.locate_file(package_name, class_name)[0]
        B_content = self.locate_file(center_package, center_class)[0]
        class_contents = [extract_class_code(code, name) for code, name in [[A_content, class_name],[B_content, center_class]]]
        if is_class_a_calling_class_b(class_contents[0], center_class) or is_class_a_calling_class_b(class_contents[1], class_name):
            return True
        else:
            return False
    
    def find_two_cls_use_members(self, member_pair):
        #classcode, member_list  #0 is Question
        use_member_map = [{},{}]       
        for i in range(2):
            code = member_pair[0][0]
            code_lines = code.splitlines()
            member_function_list = member_pair[1][1]
    

            for j in range(len(member_function_list)):
                method_name, method_text, return_text = member_function_list[j]
                if method_name == "member_variable":
                    for x in method_text.split("\n"):
                        variable = x.split(" =")[0].split("=")[0].split(";")[0].split(" ")[-1]
                        if code.find(variable)!=-1:
                            use_line = find_x_in_lines(variable, code_lines)
                            use_member_map[i][j] = use_line
                        break
                else:
                    if code.find(method_name)!=-1:
                        use_line = find_x_in_lines(method_name, code_lines)
                        use_member_map[i][j] = use_line
            member_pair = [member_pair[1], member_pair[0]]
        return use_member_map
    def get_members(self, package_name, class_name):
        content = self.locate_file(package_name, class_name)[0]
        # print(content)
        class_contents = extract_class_code(content, class_name)
        METHOD_REGEX = re.compile(r'(\w+)\s+(\w+)\s*\(.*\)\s*\{')
        method_matches = METHOD_REGEX.findall(class_contents)
        res = []
        for return_type, method_name in method_matches:
            method_code = extract_function_code(class_contents, method_name)
            return_line = find_x_in_lines("return", method_code)
            res.append([method_name, method_code, return_line])
        res.append(["member_variable",extract_member_variables(class_contents),""])
        return [class_contents, res]
    
    def chat(self, conversation_model, class_templates, rounds):
        Question = "Please select the starting class of Structure Tool from Class Code Information.\n"
        response = conversation_model.multi_chat(Question, delete_rounds = rounds)
        cls_template = class_templates.find_same_cls_name_template(response)
        WrongAnswer = "Choosed an illegal starting class."
        if cls_template is None:
            conversation_model.multi_chat(WrongAnswer, delete_rounds = rounds)
            return
        if self.locate_file(cls_template.package_name, cls_template.class_name)[0] is None:
            conversation_model.multi_chat(WrongAnswer, delete_rounds = rounds)
            return
        key = cls_template.package_name + "!" + cls_template.class_name
        
        Question = "Please select the direction of the choosing class to locate: "
        if cls_template.class_path.st() == "Question Class":
            Question += "(father, son, or brother)\n"
        elif cls_template.class_path.st().find("father")!=-1:
            Question += "father"
        elif cls_template.class_path.st().find("son")!=-1:
            Question += "son"
        elif cls_template.class_path.st().find("brother")!=-1:
            return 
        
        response = conversation_model.multi_chat(Question, delete_rounds = rounds)
        tool_generate_templates = ClassTemplateList()
        if response.find("father")!=-1:
            if key in self.fathers:
                for father_key in self.fathers[key]:
                    fa_package, fa_class = father_key.split("!")
                    new_cls_template = ClassTemplate(fa_class, fa_package, ClassPath("father", cls_template.class_path.st()))     
                    class_templates.insert(new_cls_template, rounds)  
                    tool_generate_templates.insert(new_cls_template, rounds)
        elif response.find("son")!=-1:
            if key in self.sons:
                for son_key in self.sons[key]:
                    son_package, son_class = son_key.split("!")
                    new_cls_template = ClassTemplate(son_class, son_package, ClassPath("son", cls_template.class_path.st()))     
                    class_templates.insert(new_cls_template, rounds)  
                    tool_generate_templates.insert(new_cls_template, rounds)
        elif response.find("brother")!=-1:
            for bro in self.locate_same_package_class(cls_template.package_name, cls_template.class_name):
                bro_package, bro_class = bro[1]
                new_cls_template = ClassTemplate(bro_class, bro_package, ClassPath("brother", cls_template.class_path.st()))     
                class_templates.insert(new_cls_template, rounds)  
                tool_generate_templates.insert(new_cls_template, rounds)
        if len(tool_generate_templates.class_templates) > 0:
            generate_info = tool_generate_templates.output()
            generate_info = "\n".join([
                "<Class Code Information>",
                generate_info,
                "</Class Code Information>"])
            template = "The result of using Structural Tool in this round is:\n" +generate_info
            conversation_model.multi_chat(template, delete_rounds = rounds)
            
def find_x_in_lines(x,lines):
    use_line = ""
    for line in lines:
        if x in line:
            use_line = line.strip()
    return use_line