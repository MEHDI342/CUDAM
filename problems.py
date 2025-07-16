import os
import subprocess
import json
from pathlib import Path

def run_pylint(project_dir):
    """
    Runs pylint on the specified project directory and returns the JSON output.
    """
    try:
        # Run pylint with JSON output
        result = subprocess.run(
            ['pylint', project_dir, '--output-format=json'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )

        if result.stderr:
            print("Pylint encountered an error:")
            print(result.stderr)
            # Continue processing even if pylint reports errors (like syntax errors)

        # Parse JSON output
        pylint_output = json.loads(result.stdout)
        return pylint_output

    except FileNotFoundError:
        print("Pylint is not installed or not found in the system PATH.")
        return None
    except json.JSONDecodeError:
        print("Failed to parse pylint output. Ensure pylint is producing valid JSON.")
        return None

def extract_errors(pylint_output):
    """
    Extracts only error and fatal issues from pylint output.

    Args:
        pylint_output (list): The JSON-parsed output from pylint.

    Returns:
        list: Filtered list of error issues.
    """
    error_issues = [
        {
            'File': issue.get('path', ''),
            'Line': issue.get('line', ''),
            'Column': issue.get('column', ''),
            'Symbol': issue.get('symbol', ''),
            'Message': issue.get('message', ''),
            'Type': issue.get('type', '')
        }
        for issue in pylint_output
        if issue.get('type', '').lower() in ['error', 'fatal'] and issue.get('message-id', '').startswith(('E', 'F'))
    ]

    return error_issues

def main():
    # Define your project directory
    project_dir = Path(r'C:\Users\PC\Desktop\Megie\CUDAM\CUDAM')

    if not project_dir.exists():
        print(f"The directory {project_dir} does not exist.")
        return

    print(f"Running pylint on {project_dir}...")

    pylint_output = run_pylint(str(project_dir))

    if pylint_output is None:
        print("No pylint output to process.")
        return

    relevant_errors = extract_errors(pylint_output)

    print("\n=== Pylint Errors ===")
    if relevant_errors:
        for issue in relevant_errors:
            print(f"{issue['File']}:{issue['Line']}:{issue['Column']} - {issue['Message']} [{issue['Symbol']}] ({issue['Type'].capitalize()})")
    else:
        print("No errors found.")

    # Optionally, save the results to a file
    save_results = True  # Set to False if you don't want to save
    if save_results:
        errors_file = project_dir / 'pylint_errors.txt'

        with open(errors_file, 'w', encoding='utf-8') as f:
            for issue in relevant_errors:
                f.write(f"{issue['File']}:{issue['Line']}:{issue['Column']} - {issue['Message']} [{issue['Symbol']}] ({issue['Type'].capitalize()})\n")

        print(f"\nErrors saved to {errors_file}")

if __name__ == "__main__":
    main()
