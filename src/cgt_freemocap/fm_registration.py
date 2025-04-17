import bpy
import logging
from . import fm_interface
from . import fm_operators
from ..cgt_core.cgt_utils import cgt_user_prefs


modules = [
    fm_operators,
    fm_interface
]


FM_ATTRS = {
    "load_raw": False,
    "quickload": False,
}


@bpy.app.handlers.persistent
def save_preferences(*args):
    user = bpy.context.scene.modernar_freemocap_settings  # noqa: Updated name
    cgt_user_prefs.set_prefs(**{attr: getattr(user, attr, default) for attr, default in FM_ATTRS.items()})


@bpy.app.handlers.persistent
def load_preferences(*args):
    stored_preferences = cgt_user_prefs.get_prefs(**FM_ATTRS)
    user = bpy.context.scene.modernar_freemocap_settings # noqa: Updated name
    for property_name, value in stored_preferences.items():
        # Check if the attribute exists on the PropertyGroup before setting
        if hasattr(user, property_name):
            try:
                setattr(user, property_name, value)
            except Exception as e:
                logging.warning(f"Could not set {property_name} on {user}: {e}")
        else:
            logging.warning(f"{property_name} - not available on {user}.")


def register():
    for module in modules:
        module.register()

    bpy.app.handlers.save_pre.append(save_preferences)
    bpy.app.handlers.load_post.append(load_preferences)


def unregister():
    for module in reversed(modules):
        module.unregister()

