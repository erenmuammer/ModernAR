import sys
import os
from pathlib import Path

# Add vendor directory to sys.path
VENDOR_DIR = Path(__file__).parent.parent / "vendor"
if VENDOR_DIR.is_dir():
    # Add vendor directory first to avoid potential conflicts with system packages
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
else:
    print(f"ModernAR Warning: Vendor directory not found at {VENDOR_DIR}")


import importlib
import logging
# sys is already imported above
# from typing import Optional # This seems unused in the snippet shown, can be kept or removed based on full file context
# from pathlib import Path # Path is already imported above

# Ensure all modules are reloaded from new files,
# when the addon is removed and a new version is installed in the same session,
# or when Blender's 'Reload Scripts' operator has been called.


PACKAGE_PATH = Path(__file__).parent.parent.parent
PACKAGE_NAME = PACKAGE_PATH.name


def import_module(module):
    importlib.import_module(f"{PACKAGE_NAME}{module}")


def reload_module(module):
    importlib.reload(sys.modules[f"{PACKAGE_NAME}{module}"])


def get_reload_list(sub_dirs):
    reload_list = []

    for sub_dir in sub_dirs:
        files = [p for p in sub_dir.rglob(
            "*.py") if not p.stem.startswith('_')]
        for file in files:
            parents = get_parents(file, [])
            imp_path = ""
            for parent in reversed(parents):
                imp_path += f".{parent}"
            imp_path += f".{file.stem}"
            reload_list.append(imp_path)
    return reload_list


def get_parents(file: Path, parents: list):
    if file.parent.name != PACKAGE_NAME:
        parents.append(file.parent.name)
        get_parents(file.parent, parents)
    return parents


def manage_imports(dirs: Optional[list] = None):
    if dirs is None:
        s = [PACKAGE_PATH / 'src']
    else:
        s = [PACKAGE_PATH / d for d in dirs]

    reload_list = get_reload_list(s)
    for module in reload_list:
        reload = True
        try:
            import_module(module)
        except (ModuleNotFoundError, ImportError) as e:
            reload = False
            logging.error(f"Import {module} failed: {e}")

        if reload:
            reload_module(module)
