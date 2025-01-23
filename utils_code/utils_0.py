import json
import os
def read_task_in_jsonl(task_file_path):
    result = []
    with open(task_file_path,'r',encoding = "utf8") as f:
        data_list = f.readlines()
        for data in data_list:

            x = json.loads(data)
            result.append(x)
    return result

def output_jsonl(jsonl, output_file):
    with open(output_file, 'w', encoding='utf-8') as output_file:
        for entry in jsonl:
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write('\n')

def split_smell(split_smells):
    result = []
    for smell in split_smells:
        for k in range(split_smells[smell]):
            result.append([smell, split_smells[smell], k])
    return result

def merge_smell(split_smells, output_path):
    for smell in split_smells:
        v = split_smells[smell]
        result = []
        for i in range(v):
            result += read_task_in_jsonl(os.path.join(output_path, smell + "_"+str(i)+"_"+str(v)+".jsonl"))
        output_jsonl(result, os.path.join(output_path, smell+".jsonl"))


useful_design_smell = {
    "Broken Hierarchy":1,
    "Cyclic Hierarchy":1,
    "Cyclically-dependent Modularization":1,
    "Multipath Hierarchy":1,
    "Wide Hierarchy":1,
    "Deep Hierarchy":1,
    "Feature Envy":1,
    "Rebellious Hierarchy":1
    # "Duplicate Abstraction":1
}

smell_description_dic = {
    "Broken Hierarchy": 'Definition: A code smell that occurs when a class in a hierarchy violates the expected "is-a" relationship or inheritance contract, leading to inconsistent or unexpected behavior.\n'    +"Description: Broken hierarchy often manifests when a subclass overrides or ignores significant parts of its parent class's functionality, making it behave in ways not aligned with the parent class. This violates the Liskov Substitution Principle, which states that a subclass should be substitutable for its parent class.\n",
    
    "Cyclic Hierarchy":"Definition: A code smell where dependencies within a hierarchy form a circular chain, creating interdependent relationships.\n"+"Description: Cyclic hierarchies occur when two or more classes in an inheritance chain depend on each other directly or indirectly. This makes the hierarchy hard to understand, test, and maintain, as changes in one class can propagate unpredictably through the cycle.\n",
    
    "Cyclically-dependent Modularization":"Definition: A modularization issue where two or more modules depend on each other circularly, leading to tightly coupled components.\n"+"Description: This problem violates modularity principles by creating a strong dependency loop, reducing the system's flexibility. Refactoring cyclic dependencies is crucial for achieving better modularity, as it simplifies testing, maintenance, and reuse of modules.\n",
    
    "Multipath Hierarchy":"Definition: A code smell where a class inherits from multiple paths in a complex inheritance hierarchy.\n"+"Description: This typically occurs in multiple inheritance scenarios where a class inherits from two or more classes that share a common ancestor. Multipath hierarchies can lead to ambiguity (e.g., the diamond problem) and increased complexity in understanding the behavior of the derived class.\n",
    
    "Wide Hierarchy":"Definition: A code smell where a class has too many immediate subclasses, making the hierarchy excessively wide.\n"+"Description: A wide hierarchy makes it harder to understand the relationship between subclasses and their parent class. It can also indicate poor abstraction, where a single class tries to act as the parent for too many disparate entities.\n",
    
    "Deep Hierarchy":"Definition: A code smell where the inheritance chain is excessively long, making the hierarchy deep and difficult to navigate.\n"+"Description: A deep hierarchy complicates understanding and maintaining the code, as developers must traverse multiple levels of inheritance to grasp how a particular class behaves. This can also lead to fragility, as changes in a parent class may have unexpected ripple effects.\n",
    
    "Feature Envy":"Definition: A code smell where a method of one class is overly dependent on the data or methods of another class.\n"+"Description: Feature envy violates the principle of encapsulation by allowing a method to focus on another class's internal details rather than its own. Refactoring to move the method to the class it envies often resolves this issue and improves cohesion.\n",
    
    "Rebellious Hierarchy":"Definition: A code smell where subclasses in a hierarchy resist following the expected behavior defined by their parent class.\n"+"Description: Rebellious hierarchies occur when subclasses override parent class methods in ways that break the expected pattern or contract. This makes the hierarchy unpredictable and undermines polymorphism, leading to fragile designs.\n",
    
    "Duplicate Abstraction":"Definition: A code smell where similar abstractions are repeated across the codebase rather than being unified.\n"+"Description: Duplicate abstraction indicates redundant code, typically because the same concept is implemented multiple times in slightly different ways. This increases maintenance effort and makes the system more prone to bugs. Extracting a common abstraction can eliminate the duplication.\n"

}

smell_example_dic = {
    "Cyclically-dependent Modularization":"Example of Cyclically-dependent Modularization:\nImagine a system with three modules: Module A, Module B, and Module C. These modules have the following relationships:\n\nModule A depends on Module B.\nModule B depends on Module C.\nModule C depends back on Module A.\nThis creates a circular dependency among the three modules, illustrated as:\n\nModule A → Module B → Module C → Module A"
}