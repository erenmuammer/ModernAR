bl_info = {
    "name":        "ModernAR",
    "description": "ModernAR",
    "author":      "Muammer Eren",
    "version":     (1, 6, 1),
    "blender":     (3, 0, 0),
    "location":    "3D View > Tool",
    "support":     "COMMUNITY",
    "category":    "Animation"
}


def reload_modules():
    from .src import cgt_imports
    cgt_imports.manage_imports()


def register():
    from .src import cgt_registration
    from . import preferences
    preferences.register()
    cgt_registration.register()


def unregister():
    from .src import cgt_registration
    from . import preferences
    cgt_registration.unregister()
    preferences.unregister()


if __name__ == '__main__':
    from src.cgt_core.cgt_utils import cgt_logging
    # cgt_logging.init('')
    register()
