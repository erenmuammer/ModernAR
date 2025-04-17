import bpy
from .. import cgt_naming
from pathlib import Path

# Simplified addon_dir_name for bl_idname
# Assuming the addon directory name is the parent of 'src'
addon_package_name = Path(__file__).parent.parent.parent.parent.name


class DefaultPanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ModernAR"
    bl_options = {"HEADER_LAYOUT_EXPAND"}


class PT_UI_CGT_Panel(DefaultPanel, bpy.types.Panel):
    bl_label = cgt_naming.ADDON_NAME
    bl_idname = "UI_PT_CGT_Panel"

    def draw(self, context):
        layout = self.layout
        
        # Hand gesture detection start/stop buttons
        row = layout.row(align=True)
        row.operator("cgt.start_hand_tracking", text="Start Hand Tracking", icon="PLAY")
        row.operator("cgt.stop_hand_tracking", text="Stop", icon="PAUSE")
        
        # Smoothing factor
        row = layout.row()
        row.prop(context.scene, "cgt_smoothing", text="Smoothing Factor", slider=True)


classes = [
    PT_UI_CGT_Panel,
]

_registered_classes = [] # Keep track of what this module registered

def register():
    global _registered_classes
    _registered_classes = [] # Reset on register attempt

    # Register property first
    if not hasattr(bpy.types.Scene, 'cgt_smoothing'):
        try:
            bpy.types.Scene.cgt_smoothing = bpy.props.FloatProperty(
                name="Smoothing Factor",
                description="Hand gesture smoothing factor",
                default=0.3,
                min=0.0,
                max=1.0,
                step=0.1
            )
        except Exception as e:
             print(f"Failed to register property 'cgt_smoothing': {e}")
    # else: # Property already exists, do nothing or log
    #    print("Property 'cgt_smoothing' already exists.")

    # Register classes
    for cls in classes:
        if not hasattr(bpy.types, cls.__name__): # Check if NOT already registered by name
            try:
                bpy.utils.register_class(cls)
                _registered_classes.append(cls) # Add to our list only if successful
            except Exception as e:
                print(f"Failed to register class {cls.__name__}: {e}")
        # else: # Class already registered, do nothing or log
        #    print(f"Class {cls.__name__} appears to be already registered.")


def unregister():
    global _registered_classes

    # Unregister classes that *this module* registered, in reverse order
    for cls in reversed(_registered_classes):
         try:
             bpy.utils.unregister_class(cls)
         except RuntimeError as e: # Catch potential errors during unregistration
             print(f"Error unregistering class {cls.__name__}: {e}")
         except Exception as e:
             print(f"Unexpected error unregistering class {cls.__name__}: {e}")
    _registered_classes = [] # Clear the list

    # Unregister property - Original robust check is likely best
    if hasattr(bpy.types.Scene, 'cgt_smoothing'):
        try:
            # Check if the property is actually owned/set by our addon might be needed
            # For now, assume if it exists, we might need to remove it on unregister
            del bpy.types.Scene.cgt_smoothing
        except Exception as e:
            print(f"Failed to delete scene property 'cgt_smoothing': {e}")
