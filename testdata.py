import os
import re

def generate_project_structure(directory, indent_level=0):
    structure = ""
    for root, dirs, files in os.walk(directory):
        if 'venv' in root:
            continue

        level = root.replace(directory, '').count(os.sep)
        indent = '│   ' * (level - indent_level)
        structure += f"{indent}├── {os.path.basename(root)}/\n"
        sub_indent = '│   ' * (level + 1 - indent_level)
        for file in files:
            if file.endswith('.py'):
                structure += f"{sub_indent}├── {file}\n"
        dirs[:] = [d for d in dirs if d != 'venv']  # Skip venv directory

    return structure

def extract_classes_and_methods(content):
    class_regex = r'class\s+(\w+)\s*(\(.*?\))?:'
    method_regex = r'def\s+(\w+)\s*\(.*?\):'

    extracted_content = ""
    class_matches = re.findall(class_regex, content)

    for class_match in class_matches:
        class_name = class_match[0]
        extracted_content += f"\nClass: {class_name}\n"
        extracted_content += "-" * 80 + "\n"

        method_matches = re.findall(method_regex, content)
        for method_match in method_matches:
            extracted_content += f"  Method: {method_match}\n"

    return extracted_content

def read_files_recursively(directory):
    content = ""
    for root, dirs, files in os.walk(directory):
        if 'venv' in root:
            continue

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                content += f"File: {file_path}\n\n"
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        content += file_content

                        extracted_classes_methods = extract_classes_and_methods(file_content)
                        content += extracted_classes_methods

                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='ISO-8859-1') as f:
                            file_content = f.read()
                            content += file_content
                    except Exception as e:
                        content += f"Error reading file: {e}"
                content += "\n\n" + "-"*80 + "\n\n"
    return content

def save_content_to_txt(directory, output_file):
    print("Starting the process...")
    project_structure = generate_project_structure(directory)
    file_content = read_files_recursively(directory)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Project Structure:\n\n")
        f.write(project_structure)
        f.write("\n\n" + "="*80 + "\n\n")
        f.write(file_content)
    print("Process completed successfully.")

# Usage
project_directory = r"C:\Users\PC\Desktop\Megie\CUDAM\CUDAM"
output_file = r"C:\Users\PC\Desktop\Megie\CUDAM\CUDAM\projett_content.txt"

try:
    save_content_to_txt(project_directory, output_file)
except PermissionError:
    print("Permission denied. Please check your write permissions or choose a different output location.")
except Exception as e:
    print(f"An error occurred: {e}")
