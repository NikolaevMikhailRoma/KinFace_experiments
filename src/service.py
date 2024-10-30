import os
from typing import Dict, List
# from config import DATASET_FOLDER


def show_project_structure() -> None:
    """
    Shows all file and directory paths in the project structure,
    excluding hidden directories and service folders.
    """
    project_root = os.path.dirname(os.getcwd())

    print("\n=== Project Structure ===")

    for root, dirs, files in os.walk(project_root):
        # Skip hidden and service directories
        if any(part.startswith('.') for part in root.split(os.sep)) or \
                'venv' in root or '__pycache__' in root:
            continue

        # Get and print relative path for directory
        rel_path = os.path.relpath(root, project_root)
        if rel_path == '.':
            print("📁 project_root")
        else:
            print(f"📁 {rel_path}")

        # Print paths for all non-hidden files
        for file in files:
            if not file.startswith('.'):
                file_path = os.path.join(rel_path, file)
                print(f"📄 {file_path}")


# def show_dataset_structure(max_files: int = 10) -> None:
#     """
#     Analyzes and prints the dataset structure, including nested directories.
#
#     Args:
#         max_files: Maximum number of files to display per directory
#     """
#     print("\n=== Dataset Structure ===")
#     print(f"Base path: {DATASET_FOLDER}")
#
#     try:
#         total_files = 0
#         total_dirs = 0
#
#         for root, dirs, files in os.walk(DATASET_FOLDER):
#             # Get relative path
#             rel_path = os.path.relpath(root, DATASET_FOLDER)
#             if rel_path == '.':
#                 rel_path = 'dataset_root'
#
#             # Filter visible files
#             visible_files = sorted([f for f in files if not f.startswith('.')])
#
#             # Update statistics
#             total_files += len(visible_files)
#             if root != DATASET_FOLDER:
#                 total_dirs += 1
#
#             # Print directory info if it contains files
#             if visible_files:
#                 print(f"\nDirectory: {rel_path}")
#                 print(f"Total files in directory: {len(visible_files)}")
#                 print("Example files:")
#                 for file in visible_files[:max_files]:
#                     print(f"  - {file}")
#                 if len(visible_files) > max_files:
#                     print(f"  ... and {len(visible_files) - max_files} more files")
#
#         print(f"\nTotal Statistics:")
#         print(f"Total directories: {total_dirs}")
#         print(f"Total files: {total_files}")
#
#     except FileNotFoundError:
#         print(f"Error: Dataset directory not found: {DATASET_FOLDER}")
#     except Exception as e:
#         print(f"Error analyzing dataset: {str(e)}")


if __name__ == "__main__":
    show_project_structure()
    # show_dataset_structure()