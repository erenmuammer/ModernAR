import bpy


class MP_PG_Properties(bpy.types.PropertyGroup):
    # region mediapipe props
    enum_detection_type: bpy.props.EnumProperty(
        name="Target",
        description="Select detection type tracking.",
        items=(
            ("HAND", "Hands", ""),
            ("FACE", "Face", ""),
            ("POSE", "Pose", ""),
            ("HOLISTIC", "Holistic", ""),
        )
    )

    refine_face_landmarks: bpy.props.BoolProperty(
        name="Refine Face Landmarks", default=False,
        description="Whether to further refine the landmark coordinates "
                    "around the eyes and lips, and output additional landmarks "
                    "around the irises by applying the Attention Mesh Model. "
                    "Default to false.")

    # downloading during session seem inappropriate (therefor max 1)
    holistic_model_complexity: bpy.props.IntProperty(
        name="Model Complexity", default=1, min=0, max=1,
        description="Complexity of the pose landmark model: "
                    "0, 1 or 1. Landmark accuracy as well as inference "
                    "latency generally go up with the model complexity. "
                    "Default to 1.")

    # downloading during session seem inappropriate (therefor max 1)
    pose_model_complexity: bpy.props.IntProperty(
        name="Model Complexity", default=1, min=0, max=1,
        description="Complexity of the pose landmark model: "
                    "0, 1 or 1. Landmark accuracy as well as inference "
                    "latency generally go up with the model complexity. "
                    "Default to 1.")

    hand_model_complexity: bpy.props.IntProperty(
        name="Model Complexity", default=1, min=0, max=1,
        description="Complexity of the hand landmark model: "
                    "0 or 1. Landmark accuracy as well as inference "
                    "latency generally go up with the model complexity. "
                    "Default to 1.")

    min_detection_confidence: bpy.props.FloatProperty(
        name="Min Tracking Confidence", default=0.5, min=0.0, max=1.0,
        description="Minimum confidence value ([0.0, 1.0]) from the detection "
                    "model for the detection to be considered successful. Default to 0.5.")
    # endregion

    # region stream props
    mov_data_path: bpy.props.StringProperty(
        name="File Path",
        description="File path to .mov file.",
        default='*.mov;*mp4',
        options={'HIDDEN'},
        maxlen=1024,
        subtype='FILE_PATH'
    )

    enum_stream_type: bpy.props.EnumProperty(
        name="Stream Backend",
        description="Sets Stream backend.",
        items=(
            ("0", "default", ""),
            ("1", "capdshow", "")
        )
    )

    enum_stream_dim: bpy.props.EnumProperty(
        name="Stream Dimensions",
        description="Dimensions for video Stream input.",
        items=(
            ("sd", "720x480 - recommended", ""),
            ("hd", "1240x720 - experimental", ""),
            ("fhd", "1920x1080 - experimental", ""),
        )
    )

    detection_input_type: bpy.props.EnumProperty(
        name="Type",
        description="Select input type.",
        items=(
            ("movie", "Movie", ""),
            ("stream", "Webcam", ""),
        )
    )

    webcam_input_device: bpy.props.IntProperty(
        name="Webcam Device Slot",
        description="Select Webcam device.",
        min=0,
        max=4,
        default=0
    )

    key_frame_step: bpy.props.IntProperty(
        name="Key Step",
        description="Select keyframe step rate.",
        min=1,
        max=12,
        default=4
    )
    # endregion

    modal_active: bpy.props.BoolProperty(
        name="modal_active",
        description="Check if operator is running",
        default=False
    )

    local_user: bpy.props.BoolProperty(
        name="Local user",
        description="Install to local user and not to blenders python site packages.",
        default=False,
    )


classes = [
    MP_PG_Properties,
]

_registered_classes = [] # Keep track of what this module registered
_property_registered = False # Keep track of the property

def register():
    global _registered_classes, _property_registered
    _registered_classes = [] # Reset on register attempt
    _property_registered = False

    # Register classes first (PointerProperty needs the type registered)
    for cls in classes:
        if not hasattr(bpy.types, cls.__name__): # Check if NOT already registered by name
            try:
                bpy.utils.register_class(cls)
                _registered_classes.append(cls) # Add to our list only if successful
            except Exception as e:
                print(f"Failed to register class {cls.__name__}: {e}")
        else:
            # If already registered, assume it might be ours, add to list for unregister attempt
            # A more robust system might check __module__ but this is simpler
            _registered_classes.append(cls) 
            # print(f"Class {cls.__name__} appears to be already registered.")

    # Register PointerProperty
    if not hasattr(bpy.types.Scene, 'modernar_mediapipe_settings'):
        try:
            bpy.types.Scene.modernar_mediapipe_settings = bpy.props.PointerProperty(type=MP_PG_Properties)
            _property_registered = True
        except Exception as e:
             print(f"Failed to register PointerProperty 'modernar_mediapipe_settings': {e}")
    # else: # Property already exists
    #     print("PointerProperty 'modernar_mediapipe_settings' already exists.")


def unregister():
    global _registered_classes, _property_registered

    # Unregister PointerProperty first (classes might be referenced)
    if hasattr(bpy.types.Scene, 'modernar_mediapipe_settings'):
        # Optional: Check if _property_registered is True before deleting?
        # Might prevent issues if another script registered it.
        try:
            del bpy.types.Scene.modernar_mediapipe_settings
            _property_registered = False
        except Exception as e:
            print(f"Failed to delete scene property 'modernar_mediapipe_settings': {e}")

    # Unregister classes that *this module* tried to register, in reverse order
    for cls in reversed(_registered_classes):
         try:
             # We attempt unregister even if we didn't register it, 
             # because if it failed registration before, it might exist from a previous addon load.
             # A more complex system could track success/failure better.
             bpy.utils.unregister_class(cls)
         except RuntimeError:
             # This often means it was already unregistered, ignore.
             pass
         except Exception as e:
             print(f"Unexpected error unregistering class {cls.__name__}: {e}")
    _registered_classes = [] # Clear the list
