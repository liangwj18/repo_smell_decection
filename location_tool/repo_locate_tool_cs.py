from .repo_locate_tool import repo_locate_tool, Recursive_Traversal_Files
import os

class repo_locate_tool_cs(repo_locate_tool):
    def __init__(self, repo_local_path):
        super().__init__(repo_local_path)
        self.suffix = ".cs"
    def collect_all_repo_files(self):
        self.all_repo_files = Recursive_Traversal_Files(self.repo_local_path, '.cs')
        self.all_repo_files = [[x.split("/")[-1].split(".cs")[0], x] for x in self.all_repo_files]
        self.all_repo_files.sort(key = lambda x: x[0])
    def check_package_class(file_content, package_name, class_name):
        # if file_content.find("package "+package_name) != -1 and file_content.find("class "+class_name) != -1:
        #     return True
        # else:
        #     return False
        pass