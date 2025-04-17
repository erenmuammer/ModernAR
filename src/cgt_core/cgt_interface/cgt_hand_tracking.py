import bpy
import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Load MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Constants
LANDMARK_COUNT = 21
DEFAULT_SMOOTHING = 0.3
MAX_RETRY_ATTEMPTS = 3 # Max attempts to reconnect camera
RETRY_DELAY = 1.0      # Delay in seconds between retries

class CGT_OT_StartHandTracking(bpy.types.Operator):
    """Starts the hand tracking process using the webcam."""
    bl_idname = "cgt.start_hand_tracking"
    bl_label = "Start Hand Tracking" # English Label
    bl_description = "Starts the hand gesture detection" # English Description
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _cap = None
    _hands = None
    _is_running = False
    _previous_landmarks_dict = {} # Store previous landmarks for smoothing (per hand)
    _velocity_dict = {}           # Store velocity data for smoothing (per hand)
    _frame_count = 0
    _start_time = 0
    _fps = 0
    _retry_count = 0              # Camera reconnection retry counter

    def modal(self, context, event):
        """Main loop for capturing and processing camera frames."""
        if event.type == 'TIMER':
            if self._is_running:
                try:
                    # --- Camera Check and Reconnection ---
                    if not self._cap or not self._cap.isOpened():
                        print("Camera connection lost.")
                        if self._retry_count < MAX_RETRY_ATTEMPTS:
                            print(f"Attempting to reconnect... ({self._retry_count + 1}/{MAX_RETRY_ATTEMPTS})")
                            self._retry_count += 1
                            time.sleep(RETRY_DELAY)
                            if not self.initialize_camera():
                                print("Reconnection attempt failed.")
                                return {'PASS_THROUGH'} # Keep modal running, retry next timer tick
                            else:
                                print("Camera reconnected successfully.")
                                self._retry_count = 0 # Reset retry count on success
                        else:
                            self.report({'ERROR'}, "Camera connection failed after multiple attempts. Stopping.")
                            self.cleanup()
                            return {'FINISHED'}
                        return {'PASS_THROUGH'} # Skip processing this frame if camera was just reconnected or failed

                    # --- Frame Capture ---
                    success, frame = self._cap.read()
                    if not success:
                        # Handle read error (might indicate camera disconnect)
                        print("Warning: Could not read frame from camera.")
                        # Trigger reconnection logic on next tick
                        if self._cap:
                           self._cap.release()
                        self._cap = None
                        return {'PASS_THROUGH'}

                    # Reset retry count if frame read is successful after previous issues
                    if self._retry_count > 0:
                         print("Camera read successful, resetting retry count.")
                         self._retry_count = 0

                    # --- FPS Calculation ---
                    self._frame_count += 1
                    elapsed_time = time.time() - self._start_time
                    if elapsed_time >= 1.0:
                        self._fps = self._frame_count / elapsed_time
                        self._frame_count = 0
                        self._start_time = time.time()

                    # --- Image Processing ---
                    # Flip the frame horizontally for a later selfie-view display
                    frame = cv2.flip(frame, 1)
                    # Convert the BGR image to RGB
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Process the image and find hands
                    results = self._hands.process(image_rgb)

                    # --- Display Information on Frame ---
                    # Display FPS
                    cv2.putText(frame, f'FPS: {int(self._fps)}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Display number of detected hands
                    num_hands = 0
                    if results.multi_hand_landmarks:
                        num_hands = len(results.multi_hand_landmarks)
                        cv2.putText(frame, f'Detected Hands: {num_hands}',
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, 'No Hands Detected',
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # --- Hand Landmark Processing ---
                    if results.multi_hand_landmarks:
                        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                            # Determine hand type (Right/Left)
                            # Mediapipe's 'label' can be 'Left' or 'Right'
                            hand_type = handedness.classification[0].label
                            hand_id = f"{hand_type}_{idx}" # Unique ID for each detected hand

                            # Display hand type on the frame
                            h, w, _ = frame.shape
                            # Use wrist landmark (index 0) position for text placement
                            cx = int(hand_landmarks.landmark[0].x * w)
                            cy = int(hand_landmarks.landmark[0].y * h)
                            cv2.putText(frame, hand_type, (cx - 30, cy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                            # Extract landmark coordinates
                            hand_points = []
                            valid_landmarks = True
                            for landmark in hand_landmarks.landmark:
                                if hasattr(landmark, 'x') and hasattr(landmark, 'y') and hasattr(landmark, 'z'):
                                    hand_points.append({
                                        'x': float(landmark.x),
                                        'y': float(landmark.y),
                                        'z': float(landmark.z)
                                    })
                                else:
                                    valid_landmarks = False
                                    break # Skip this hand if any landmark is invalid

                            # Process landmarks only if all are valid
                            if valid_landmarks and len(hand_points) == LANDMARK_COUNT:
                                # Apply advanced smoothing
                                smoothed_points = self.advanced_smooth(hand_points, hand_id)

                                # Update Blender hand animation
                                self.update_hand_animation(smoothed_points, hand_type)
                            elif not valid_landmarks:
                                 print(f"Warning: Invalid landmark data for hand {hand_id}.")
                            else:
                                print(f"Warning: Incomplete landmark data for hand {hand_id} ({len(hand_points)}/{LANDMARK_COUNT}).")

                            # Draw hand landmarks and connections on the frame
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Landmark color
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # Connection color
                            )

                    # --- Display Frame ---
                    cv2.imshow('ModernAR - Hand Tracking', frame) # Updated window title

                    # --- Exit Condition ---
                    # Check for 'q' key press to quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.cleanup()
                        return {'FINISHED'}

                except cv2.error as e:
                    print(f"OpenCV Error in modal loop: {e}")
                    # Specific handling for camera related errors, if possible
                    if "camera" in str(e).lower() or "source" in str(e).lower():
                         if self._cap: self._cap.release()
                         self._cap = None
                         print("Attempting recovery after OpenCV error...")
                    # For other CV errors, maybe just report and continue/stop
                    # self.report({'WARNING'}, f"OpenCV error: {e}")
                    # return {'PASS_THROUGH'} # Or stop:
                    # self.cleanup()
                    # return {'FINISHED'}

                except Exception as e:
                    print(f"Error in modal loop: {str(e)}")
                    # Optionally report non-critical errors
                    # self.report({'WARNING'}, f"Error: {e}")
                    # Attempt to continue if possible, otherwise cleanup
                    # For critical errors, uncomment below:
                    # self.report({'ERROR'}, f"Critical error: {e}. Stopping.")
                    # self.cleanup()
                    # return {'FINISHED'}

        elif event.type == 'ESC':
            # Stop if ESC key is pressed
            self.cleanup()
            return {'CANCELLED'}

        return {'PASS_THROUGH'} # Pass other events through

    def initialize_camera(self):
        """Initializes the camera connection, trying different backends/indices."""
        print("Initializing camera...")
        if self._cap is not None:
            print("Releasing existing camera capture...")
            self._cap.release()
            self._cap = None

        # List of camera indices and backends to try
        # 0, 1 are common indices. CAP_AVFOUNDATION is for macOS.
        # Add more indices or backends if needed (e.g., cv2.CAP_DSHOW for Windows)
        camera_options = [
            (0, cv2.CAP_ANY), # Default API with index 0
            (1, cv2.CAP_ANY), # Default API with index 1
            (0, cv2.CAP_AVFOUNDATION), # AVFoundation for macOS
            # ("/dev/video0", cv2.CAP_V4L2) # Example for V4L2 on Linux
        ]

        for index, backend in camera_options:
            try:
                print(f"Trying camera index: {index}, backend: {backend}...")
                temp_cap = cv2.VideoCapture(index, backend)

                if temp_cap is not None and temp_cap.isOpened():
                    # Try to grab a frame to confirm it's working
                    ret = temp_cap.grab()
                    if ret:
                        print(f"Successfully opened camera {index} with backend {backend}.")
                        self._cap = temp_cap
                        # Set camera properties (optional, might fail on some cameras)
                        try:
                             self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                             self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                             self._cap.set(cv2.CAP_PROP_FPS, 30)
                             self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce latency
                             print("Set camera properties (W:1280, H:720, FPS:30, Buffer:1)")
                        except Exception as prop_e:
                             print(f"Warning: Could not set camera properties: {prop_e}")
                        return True # Successfully initialized
                    else:
                        print("Failed to grab frame, releasing.")
                        temp_cap.release()
                else:
                     print("Failed to open.")
                     if temp_cap: temp_cap.release()

            except Exception as e:
                print(f"Error opening camera {index} with backend {backend}: {str(e)}")
                if 'temp_cap' in locals() and temp_cap is not None:
                    temp_cap.release()
                continue # Try next option

        print("Failed to initialize camera after trying all options.")
        self._cap = None
        return False # Failed to initialize

    def execute(self, context):
        """Called when the operator starts. Sets up camera and MediaPipe."""
        print("Executing StartHandTracking operator...")
        # Clean up any previous instances first
        self.cleanup()

        # --- Initialize Camera ---
        if not self.initialize_camera():
            self.report({'ERROR'}, "Failed to open camera. Check connections and permissions.")
            return {'CANCELLED'}

        # --- Initialize MediaPipe Hands ---
        print("Initializing MediaPipe Hands...")
        try:
            self._hands = mp_hands.Hands(
                static_image_mode=False,      # Process video stream
                max_num_hands=4,              # Detect up to 4 hands
                min_detection_confidence=0.7, # Higher confidence for initial detection
                min_tracking_confidence=0.5,  # Lower confidence for tracking continuity
                model_complexity=1            # 0=light, 1=full, 2=heavy
            )
        except Exception as e:
             self.report({'ERROR'}, f"Failed to initialize MediaPipe Hands: {e}")
             self.cleanup() # Release camera if hands init fails
             return {'CANCELLED'}
        print("MediaPipe Hands initialized.")

        # --- Reset State Variables ---
        self._is_running = True
        self._previous_landmarks_dict = {} # Clear previous smoothing data
        self._velocity_dict = {}           # Clear previous velocity data
        self._frame_count = 0
        self._start_time = time.time()
        self._fps = 0
        self._retry_count = 0              # Reset camera retry count

        # --- Start Modal Timer ---
        wm = context.window_manager
        # Timer interval (0.033 seconds = ~30 FPS)
        self._timer = wm.event_timer_add(1.0 / 30.0, window=context.window)
        wm.modal_handler_add(self)
        print("Modal handler added, starting tracking loop.")

        self.report({'INFO'}, "Hand tracking started. Press 'Q' in the OpenCV window or ESC in Blender to stop.")
        print("Note: If running on macOS/Linux for the first time, you might need to grant camera permissions to Blender.")

        return {'RUNNING_MODAL'}

    def cleanup(self):
        """Releases resources when the operator stops or is cancelled."""
        print("Cleaning up hand tracking resources...")
        if self._timer:
            try:
                # Access window manager safely
                wm = bpy.context.window_manager
                if wm:
                    wm.event_timer_remove(self._timer)
                    print("Timer removed.")
            except AttributeError:
                 print("Could not access window manager to remove timer (likely Blender shutting down).")
            except Exception as e:
                 print(f"Error removing timer: {e}")
            self._timer = None

        if self._cap is not None:
            print("Releasing camera capture...")
            self._cap.release()
            self._cap = None
            print("Camera released.")

        # Close OpenCV windows only if they might be open
        try:
             cv2.destroyAllWindows()
             print("OpenCV windows destroyed.")
        except cv2.error as e:
             # Ignore errors if windows were already closed or never opened
             print(f"Ignoring OpenCV destroyAllWindows error: {e}")


        if self._hands:
             # Although Hands() doesn't have an explicit close, good practice
             self._hands = None
             print("MediaPipe Hands instance cleared.")

        self._is_running = False
        self._previous_landmarks_dict = {} # Clear smoothing data
        self._velocity_dict = {}           # Clear velocity data
        self._retry_count = 0              # Reset retry count
        print("Cleanup complete.")


    def advanced_smooth(self, current_landmarks, hand_id):
        """Applies velocity-based adaptive exponential smoothing to landmarks."""
        if hand_id not in self._previous_landmarks_dict:
            # First frame for this hand_id, initialize
            self._previous_landmarks_dict[hand_id] = current_landmarks
            # Initialize velocity to zero
            self._velocity_dict[hand_id] = [{'x': 0.0, 'y': 0.0, 'z': 0.0}] * LANDMARK_COUNT
            return current_landmarks

        # Get smoothing factor from scene property (or use default)
        alpha_base = bpy.context.scene.cgt_smoothing if hasattr(bpy.context.scene, 'cgt_smoothing') else DEFAULT_SMOOTHING

        smoothed_landmarks = []
        previous_frame_landmarks = self._previous_landmarks_dict[hand_id]
        previous_velocities = self._velocity_dict[hand_id]

        for i in range(LANDMARK_COUNT):
            current_lm = current_landmarks[i]
            previous_lm = previous_frame_landmarks[i]
            prev_vel = previous_velocities[i]

            # --- 1. Calculate Current Velocity ---
            current_velocity = {
                'x': current_lm['x'] - previous_lm['x'],
                'y': current_lm['y'] - previous_lm['y'],
                'z': current_lm['z'] - previous_lm['z']
            }

            # --- 2. Smooth Velocity (Low-pass filter on velocity) ---
            # This helps stabilize the velocity estimate itself
            velocity_alpha = 0.3 # How much the current velocity influences the smoothed velocity
            smoothed_velocity = {
                'x': velocity_alpha * current_velocity['x'] + (1 - velocity_alpha) * prev_vel['x'],
                'y': velocity_alpha * current_velocity['y'] + (1 - velocity_alpha) * prev_vel['y'],
                'z': velocity_alpha * current_velocity['z'] + (1 - velocity_alpha) * prev_vel['z']
            }

            # --- 3. Calculate Adaptive Smoothing Factor ---
            # Increase smoothing factor (alpha) for faster movements
            # Reduce smoothing (lower alpha) for slower movements (reduces jitter when still)
            velocity_magnitude = math.sqrt(
                smoothed_velocity['x']**2 +
                smoothed_velocity['y']**2 +
                smoothed_velocity['z']**2
            )

            # Adapt alpha based on velocity magnitude.
            # The factor '2' here is arbitrary, adjust based on testing.
            # Higher factor means velocity has more impact on reducing smoothing.
            adaptive_alpha = min(1.0, alpha_base + velocity_magnitude * 2.0) # Clamp alpha between base and 1.0

            # --- 4. Apply Smoothed Position ---
            # Standard exponential smoothing using the adaptive alpha
            smoothed_point = {
                'x': adaptive_alpha * current_lm['x'] + (1 - adaptive_alpha) * previous_lm['x'],
                'y': adaptive_alpha * current_lm['y'] + (1 - adaptive_alpha) * previous_lm['y'],
                'z': adaptive_alpha * current_lm['z'] + (1 - adaptive_alpha) * previous_lm['z']
            }

            smoothed_landmarks.append(smoothed_point)
            # Store the smoothed velocity for the next frame
            self._velocity_dict[hand_id][i] = smoothed_velocity

        # Store the smoothed landmarks for the next frame
        self._previous_landmarks_dict[hand_id] = smoothed_landmarks
        return smoothed_landmarks


    def update_hand_animation(self, landmarks, hand_type="Right"):
        """
        Updates the rotation of hand bones in Blender based on landmarks.
        Inspired by akg/Blender Project/blender_import.py
        Args:
            landmarks: List of smoothed landmark dictionaries ({'x', 'y', 'z'}).
            hand_type: String, "Right" or "Left".
        """
        try:
            # Get the active armature, or find the first one in the scene
            armature_obj = bpy.context.active_object
            if armature_obj is None or armature_obj.type != 'ARMATURE':
                print("No active armature found. Searching scene...")
                for obj in bpy.context.scene.objects:
                    if obj.type == 'ARMATURE':
                        armature_obj = obj
                        print(f"Found armature: {armature_obj.name}")
                        break

            if armature_obj is None or armature_obj.type != 'ARMATURE':
                print("Warning: No armature found in the scene. Cannot update animation.")
                return

            # Determine bone name prefix based on hand type
            prefix = hand_type # Assumes handedness label is "Right" or "Left"

            # Find the corresponding pose bones
            hand_pose_bones = self.find_hand_bones(armature_obj, prefix)

            if not hand_pose_bones:
                print(f"Warning: Could not find sufficient bones for {prefix} hand in armature '{armature_obj.name}'. Check bone names.")
                return

            if landmarks:
                # --- Wrist Rotation (Example using Euler rotation) ---
                # Note: Directly mapping normalized coords to Euler angles is a simplification
                # and might not produce anatomically correct rotations.
                # A more robust approach involves calculating bone vectors and angles.
                wrist_lm = landmarks[0]
                if 'hand' in hand_pose_bones:
                    wrist_bone = hand_pose_bones['hand']
                    # Example: Map x,y,z to Euler angles (adjust scaling/mapping as needed)
                    # This mapping is likely incorrect and needs refinement based on rig setup.
                    wrist_bone.rotation_mode = 'XYZ' # Ensure Euler rotation mode
                    wrist_bone.rotation_euler.x = (wrist_lm['y'] - 0.5) * math.pi * 2 # Map y-coord to x-rot
                    wrist_bone.rotation_euler.y = (wrist_lm['x'] - 0.5) * math.pi * 2 # Map x-coord to y-rot (flipped?)
                    wrist_bone.rotation_euler.z = (wrist_lm['z'] - 0.5) * math.pi * 1 # Map z-coord to z-rot (less range?)

                # --- Finger Bone Rotations ---
                # Based on akg/Blender Project/blender_import.py logic
                # Maps specific landmarks to finger bone indices
                finger_mapping = {
                    # finger_name: [landmark_indices for main, middle, tip]
                    "thumb": [1, 2, 3],      # Thumb: CMC, MCP, IP (uses 1,2,3, not 4=Tip)
                    "index": [5, 6, 7],      # Index: MCP, PIP, DIP (uses 5,6,7, not 8=Tip)
                    "middle": [9, 10, 11],   # Middle: MCP, PIP, DIP (uses 9,10,11, not 12=Tip)
                    "ring": [13, 14, 15],    # Ring: MCP, PIP, DIP (uses 13,14,15, not 16=Tip)
                    "pinky": [17, 18, 19]    # Pinky: MCP, PIP, DIP (uses 17,18,19, not 20=Tip)
                }

                for finger_name, landmark_indices in finger_mapping.items():
                    for i, bone_type in enumerate(["main", "middle", "tip"]): # Corresponds to MCP, PIP, DIP/IP
                        bone_dict_key = f"{finger_name}_{bone_type}"
                        if bone_dict_key in hand_pose_bones and landmark_indices[i] < len(landmarks):
                            finger_bone = hand_pose_bones[bone_dict_key]
                            landmark = landmarks[landmark_indices[i]]
                            prev_landmark = landmarks[landmark_indices[i]-1] if i > 0 else landmarks[0] # Use wrist for first bone

                            # --- Calculate Rotation (Example: Simple Euler Mapping) ---
                            # TODO: Implement a more robust rotation calculation.
                            # This simple mapping of normalized coordinates to Euler angles
                            # is generally insufficient for accurate finger bending.
                            # Consider calculating vectors between landmarks and using aiming constraints
                            # or inverse kinematics for better results.

                            finger_bone.rotation_mode = 'XYZ'
                            # Example: Map difference from previous landmark or absolute position?
                            # This needs significant improvement.
                            finger_bone.rotation_euler.x = (landmark['y'] - 0.5) * math.pi * 1.5
                            finger_bone.rotation_euler.y = (landmark['x'] - 0.5) * math.pi * 1.5
                            finger_bone.rotation_euler.z = (landmark['z'] - 0.5) * math.pi * 1.5

                            # Force Blender to update the viewport (optional)
                            # finger_bone.keyframe_insert(data_path="rotation_euler", frame=bpy.context.scene.frame_current)

        except AttributeError as e:
             # Handle cases where scene objects or properties might not exist as expected
             print(f"AttributeError during animation update: {e}. Check scene setup.")
        except Exception as e:
            # General error catching
            print(f"Error updating hand animation: {str(e)}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging


    def find_hand_bones(self, armature_obj, prefix="Right"):
        """
        Finds hand pose bones in the given armature based on common naming conventions.
        Args:
            armature_obj: The Blender armature object.
            prefix: "Right" or "Left" string indicating the hand side.
        Returns:
            A dictionary mapping internal bone names (e.g., "thumb_main")
            to Blender PoseBone objects, or an empty dict if not found.
        """
        # Define standard and alternative bone naming patterns
        # Standard names (e.g., Mixamo, UE Mannequin style)
        standard_bone_names = {
            "hand": f"{prefix}Hand",
            "thumb_main": f"{prefix}HandThumb1", "thumb_middle": f"{prefix}HandThumb2", "thumb_tip": f"{prefix}HandThumb3",
            "index_main": f"{prefix}HandIndex1", "index_middle": f"{prefix}HandIndex2", "index_tip": f"{prefix}HandIndex3",
            "middle_main": f"{prefix}HandMiddle1", "middle_middle": f"{prefix}HandMiddle2", "middle_tip": f"{prefix}HandMiddle3",
            "ring_main": f"{prefix}HandRing1", "ring_middle": f"{prefix}HandRing2", "ring_tip": f"{prefix}HandRing3",
            "pinky_main": f"{prefix}HandPinky1", "pinky_middle": f"{prefix}HandPinky2", "pinky_tip": f"{prefix}HandPinky3"
        }

        # Alternative names (lowercase, different separators)
        alt_bone_names = {
            "hand": f"{prefix.lower()}_hand",
            "thumb_main": f"{prefix.lower()}_thumb_01", "thumb_middle": f"{prefix.lower()}_thumb_02", "thumb_tip": f"{prefix.lower()}_thumb_03",
            "index_main": f"{prefix.lower()}_index_01", "index_middle": f"{prefix.lower()}_index_02", "index_tip": f"{prefix.lower()}_index_03",
            "middle_main": f"{prefix.lower()}_middle_01", "middle_middle": f"{prefix.lower()}_middle_02", "middle_tip": f"{prefix.lower()}_middle_03",
            "ring_main": f"{prefix.lower()}_ring_01", "ring_middle": f"{prefix.lower()}_ring_02", "ring_tip": f"{prefix.lower()}_ring_03",
            "pinky_main": f"{prefix.lower()}_pinky_01", "pinky_middle": f"{prefix.lower()}_pinky_02", "pinky_tip": f"{prefix.lower()}_pinky_03"
        }
        # Add more patterns if needed...
        alt_bone_names_dots = {
            "hand": f"{prefix}.Hand",
            "thumb_main": f"{prefix}.Hand.Thumb.01", "thumb_middle": f"{prefix}.Hand.Thumb.02", "thumb_tip": f"{prefix}.Hand.Thumb.03",
            # ... etc
        }


        found_bones = {}
        pose_bones = armature_obj.pose.bones

        # Try standard names first
        for internal_name, bone_name in standard_bone_names.items():
            if bone_name in pose_bones:
                found_bones[internal_name] = pose_bones[bone_name]

        # If not enough bones found, try alternative names
        if len(found_bones) < 8: # Heuristic: Need at least hand + some fingers
            print("Standard bone names not fully matched, trying alternatives...")
            for internal_name, bone_name in alt_bone_names.items():
                 if bone_name in pose_bones and internal_name not in found_bones:
                     print(f"Found alternative bone: {bone_name} for {internal_name}")
                     found_bones[internal_name] = pose_bones[bone_name]

        # Add more checks for other naming conventions if necessary
        # if len(found_bones) < 8:
        #     print("Trying dot-separated alternatives...")
        #     # ... check alt_bone_names_dots ...


        if not found_bones:
             print(f"Warning: No hand bones found for prefix '{prefix}' in armature '{armature_obj.name}'.")
        elif len(found_bones) < 16: # Heuristic for a full hand rig (1 hand + 3*5 fingers)
             print(f"Warning: Found only {len(found_bones)} bones for prefix '{prefix}'. Some finger animations might be missing.")

        return found_bones


class CGT_OT_StopHandTracking(bpy.types.Operator):
    """Stops the running hand tracking process."""
    bl_idname = "cgt.stop_hand_tracking"
    bl_label = "Stop Hand Tracking" # English Label
    bl_description = "Stops the hand gesture detection" # English Description

    @classmethod
    def poll(cls, context):
        # Optional: Only enable the button if the operator might be running
        # This is tricky because the running instance is in the modal handler,
        # not easily accessible globally. A simple approach is to always allow stopping.
        return True

    def execute(self, context):
        """Finds and cancels the running StartHandTracking operator instance."""
        print("Executing StopHandTracking operator...")
        found_operator = False
        # Iterate through Blender's window managers and modal handlers
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D': # Or iterate all types if unsure where modal runs
                    for region in area.regions:
                         # Look for modal handlers in the window region
                         # Note: Accessing region.modal_operators might be more direct if API allows
                         # Checking region.operators might list operators triggered from UI, not necessarily running modal ones
                         # A more reliable way is needed if this fails. Often involves storing operator refs.
                         # Trying a common pattern:
                         if hasattr(context.window_manager, 'modal_handler_add'): # Check if modal exists
                            # This is indirect. A better way: store self reference from StartHandTracking?
                            # Or check active operators:
                            active_ops = context.window_manager.operators
                            for op in active_ops:
                                 if op.bl_idname == CGT_OT_StartHandTracking.bl_idname:
                                     print(f"Found running operator instance: {op}")
                                     # Call the cleanup method directly on the instance if accessible
                                     # op.cleanup() # This might not work depending on context
                                     # Standard way is via cancel
                                     # try: # <-- REMOVE THIS LINE
                                     # Need to find the specific instance and call cancel on it.
                                     # This part is complex. The simplest is often to rely on the 'ESC' key
                                     # or have StartHandTracking check a global flag.
                                     # Let's try triggering cancel via the context, assuming it targets the modal op:
                                     # This might require the StartHandTracking op to handle a specific event type
                                     # that we trigger here.
                                     # ---> Simplification: Assume StartHandTracking's ESC handler works.
                                     # ---> We just need to tell the user.

                                     # Alternative: Use a class variable flag
                                     # CGT_OT_StartHandTracking._should_stop = True # Requires StartHandTracking to check this flag

                                     # --- Safest immediate action: Report and rely on user pressing ESC/Q ---
                                     self.report({'INFO'}, "Stop requested. Press 'ESC' in Blender or 'Q' in the OpenCV window.")
                                     found_operator = True # Indicate we sent the request
                                     # We cannot directly call cancel() reliably from here in simple way.
                                     return {'FINISHED'} # Indicate the stop button was pressed


        if not found_operator:
            self.report({'INFO'}, "Hand tracking does not appear to be running.")
            return {'CANCELLED'}
        else:
             # This part is unlikely to be reached with the current logic
             return {'FINISHED'}


def register():
    """Registers the Blender operators."""
    print("Registering ModernAR Hand Tracking operators...") # Updated print message
    try:
        bpy.utils.register_class(CGT_OT_StartHandTracking)
        bpy.utils.register_class(CGT_OT_StopHandTracking)
        print("Operators registered successfully.")
    except Exception as e:
        print(f"Error registering operators: {e}")

def unregister():
    """Unregisters the Blender operators."""
    print("Unregistering ModernAR Hand Tracking operators...") # Updated print message
    try:
        # Check if class exists before unregistering to prevent errors on reload
        if hasattr(bpy.types, CGT_OT_StartHandTracking.__name__):
             bpy.utils.unregister_class(CGT_OT_StartHandTracking)
        if hasattr(bpy.types, CGT_OT_StopHandTracking.__name__):
             bpy.utils.unregister_class(CGT_OT_StopHandTracking)
        print("Operators unregistered successfully.")
    except Exception as e:
        print(f"Error unregistering operators: {e}") 