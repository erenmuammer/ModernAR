import bpy


class UI_PT_CGT_Properties_Freemocap(bpy.types.PropertyGroup):
    freemocap_session_path: bpy.props.StringProperty(
        name="Path",
        default="/Users/Scylla/Downloads/sesh_2022-09-19_16_16_50_in_class_jsm/",
        description="Directory path to freemocap session.",
        options={'HIDDEN'},
        maxlen=1024,
        subtype='DIR_PATH'
    )
    modal_active: bpy.props.BoolProperty(default=False)
    load_raw: bpy.props.BoolProperty(
        default=False, description="Loads raw session data - may not be transferred to rigs.")
    quickload: bpy.props.BoolProperty(
        default=False, description="Quickload session folder. (Freezes Blender)")


class UI_PT_CGT_Panel_Freemocap(bpy.types.Panel):
    bl_label = "Freemocap"
    bl_parent_id = "UI_PT_CGT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "BlendAR"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        if context.mode in {'OBJECT', 'POSE'}:
            return True

    def quickload_session_folder(self, user):
        if user.modal_active:
            self.layout.row().operator("wm.cgt_quickload_freemocap_operator", text="Stop Import", icon='CANCEL')
        else:
            self.layout.row().operator("wm.cgt_quickload_freemocap_operator", text="Quickload Session Folder", icon='IMPORT')

    def load_session_folder(self, user):
        if user.modal_active:
            self.layout.row().operator("wm.cgt_load_freemocap_operator", text="Stop Import", icon='CANCEL')
        else:
            self.layout.row().operator("wm.cgt_load_freemocap_operator", text="Load Session Folder", icon='IMPORT')

    def draw(self, context):
        layout = self.layout

        user = context.scene.modernar_freemocap_settings  # noqa
        layout.row().prop(user, "freemocap_session_path")
        if not user.quickload:
            self.load_session_folder(user)
        else:
            self.quickload_session_folder(user)

        self.layout.row().operator("wm.fmc_load_synchronized_videos", text="Load synchronized videos", icon='IMAGE_PLANE')
        row = layout.row()
        row.column(align=True).prop(user, "quickload", text="Quickload", toggle=True)
        if user.quickload:
            row.column(align=True).prop(user, "load_raw", text="Raw", toggle=True)
        # layout.separator()
        # layout.row().operator("wm.fmc_bind_freemocap_data_to_skeleton", text="Bind to rig (Preview)")


classes = [
    UI_PT_CGT_Properties_Freemocap,
    UI_PT_CGT_Panel_Freemocap,
]

_registered_classes = [] # Keep track of what this module registered
_property_registered = False # Keep track of the property

def register():
    global _registered_classes, _property_registered
    _registered_classes = [] # Reset on register attempt
    _property_registered = False

    # Register classes
    for cls in classes:
        if not hasattr(bpy.types, cls.__name__): # Check if NOT already registered by name
            try:
                bpy.utils.register_class(cls)
                _registered_classes.append(cls) # Add to our list only if successful
            except Exception as e:
                print(f"Failed to register class {cls.__name__}: {e}")
        else:
            # If already registered, assume it might be ours, add to list for unregister attempt
            _registered_classes.append(cls)

    # Register PointerProperty
    if not hasattr(bpy.types.Scene, 'modernar_freemocap_settings'):
        try:
            bpy.types.Scene.modernar_freemocap_settings = bpy.props.PointerProperty(type=UI_PT_CGT_Properties_Freemocap)
            _property_registered = True
        except Exception as e:
             print(f"Failed to register PointerProperty 'modernar_freemocap_settings': {e}")
    # else: # Property already exists
    #     print("PointerProperty 'modernar_freemocap_settings' already exists.")

def unregister():
    global _registered_classes, _property_registered

    # Unregister PointerProperty first
    if hasattr(bpy.types.Scene, 'modernar_freemocap_settings'):
        try:
            del bpy.types.Scene.modernar_freemocap_settings
            _property_registered = False
        except Exception as e:
            print(f"Failed to delete scene property 'modernar_freemocap_settings': {e}")

    # Unregister classes
    for cls in reversed(_registered_classes):
         try:
             bpy.utils.unregister_class(cls)
         except RuntimeError:
             pass
         except Exception as e:
             print(f"Unexpected error unregistering class {cls.__name__}: {e}")
    _registered_classes = []


if __name__ == '__main__':
    try:
        unregister()
    except RuntimeError:
        pass

    register()
