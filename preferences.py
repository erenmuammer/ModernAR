import bpy

# Import draw functions from relevant modules
# We need to find where the draw functions for preferences are defined.
# Based on the error, cgt_mp_preferences has one. Let's assume others might too.
try:
    from .src.cgt_mediapipe import cgt_mp_preferences
except ImportError:
    cgt_mp_preferences = None
    print("ModernAR Warning: Could not import cgt_mp_preferences for preferences drawing.")

# Add imports for other potential preference draw functions here
# Example:
# try:
#     from .src.cgt_transfer import cgt_tf_preferences # Check if this exists
# except ImportError:
#     cgt_tf_preferences = None
#     print("ModernAR Warning: Could not import cgt_tf_preferences...")

# try:
#     from .src.cgt_freemocap import fm_preferences # Check if this exists
# except ImportError:
#     fm_preferences = None
#     print("ModernAR Warning: Could not import fm_preferences...")


class ModernARPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__ # Use the main package name (should be 'ModernAR')

    def draw(self, context):
        layout = self.layout
        layout.label(text="ModernAR Add-on Preferences")
        
        # Call draw functions from submodules if they exist and have a draw function
        if cgt_mp_preferences and hasattr(cgt_mp_preferences, 'draw'):
            cgt_mp_preferences.draw(self, context) # Pass self and context

        # Add calls for other preference draw functions here
        # if cgt_tf_preferences and hasattr(cgt_tf_preferences, 'draw'):
        #     cgt_tf_preferences.draw(self, context)
            
        # if fm_preferences and hasattr(fm_preferences, 'draw'):
        #     fm_preferences.draw(self, context)

# Registration handled in __init__.py usually
classes = (
    ModernARPreferences,
)

_registered_classes = []

def register():
    global _registered_classes
    _registered_classes = []
    from bpy.utils import register_class
    for cls in classes:
         if not hasattr(bpy.types, cls.__name__):
             try:
                 register_class(cls)
                 _registered_classes.append(cls)
             except Exception as e:
                 print(f"Failed to register preference class {cls.__name__}: {e}")
         else:
            _registered_classes.append(cls) # Assume registered

def unregister():
    global _registered_classes
    from bpy.utils import unregister_class
    for cls in reversed(_registered_classes):
        try:
            unregister_class(cls)
        except RuntimeError:
            pass
        except Exception as e:
            print(f"Error unregistering preference class {cls.__name__}: {e}")
    _registered_classes = [] 