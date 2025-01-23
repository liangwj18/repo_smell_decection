from .repo_locate_tool_cs import repo_locate_tool_cs
from .repo_locate_tool_java import repo_locate_tool_java

def repo_locate_tool_build(repo_code_type, repo_file_path, build_connection_graph = True):
    if repo_code_type == "java":
        return repo_locate_tool_java(repo_file_path, build_connection_graph)
    elif repo_code_type == 'cs':
        return repo_locate_tool_cs(repo_file_path, build_connection_graph)