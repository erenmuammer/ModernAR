import cv2
import mediapipe as mp
import numpy as np
import os

# Load MediaPipe Hands module and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Hands model - update settings for multi-hand detection
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,  # Increase maximum number of hands
    min_detection_confidence=0.7,  # Increase detection sensitivity
    min_tracking_confidence=0.5,  # Lower tracking sensitivity for smoother tracking
    model_complexity=1  # Increase model complexity (0, 1, or 2)
)

# Exponential smoothing parameters
ALPHA = 0.3
previous_landmarks_dict = {}  # Separate smoothing for each hand

def exponential_smooth(current_landmarks, hand_id):
    """Apply exponential smoothing separately for each hand"""
    if hand_id not in previous_landmarks_dict:
        previous_landmarks_dict[hand_id] = current_landmarks
        return current_landmarks
    
    current = np.array(current_landmarks)
    previous = np.array(previous_landmarks_dict[hand_id])
    smoothed = ALPHA * current + (1 - ALPHA) * previous
    previous_landmarks_dict[hand_id] = smoothed.tolist()
    return smoothed.tolist()

def process_frame(frame):
    """Process each frame and detect hand movements"""
    # MediaPipe works with RGB format
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        # Show number of detected hands
        num_hands = len(results.multi_hand_landmarks)
        cv2.putText(frame, f'Detected Hands: {num_hands}', 
                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Process each hand
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            # Determine hand type (Right/Left)
            hand_type = "Right" if handedness.classification[0].label == "Right" else "Left"
            hand_id = f"{hand_type}_{idx}"
            
            # Get joint coordinates (x, y, z) for each hand
            hand_points = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
            
            # Apply separate smoothing for each hand
            smoothed_points = exponential_smooth(hand_points, hand_id)
            
            # Visualize smoothed points
            for point_idx, (x, y, z) in enumerate(smoothed_points):
                hand_landmarks.landmark[point_idx].x = x
                hand_landmarks.landmark[point_idx].y = y
                hand_landmarks.landmark[point_idx].z = z
            
            # Display hand type on screen
            h, w, _ = frame.shape
            cx = int(hand_landmarks.landmark[0].x * w)
            cy = int(hand_landmarks.landmark[0].y * h)
            cv2.putText(frame, hand_type, (cx - 30, cy - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw joint points on screen
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
    else:
        # Show message when no hands are detected
        cv2.putText(frame, 'No Hands Detected', 
                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

def main():
    """Main program"""
    print("Starting Hand Movement Tracking System...")
    print("Controls:")
    print("- 'q': Exit program")
    print("- 'space': Pause/Resume video")
    print("- 'r': Restart video")
    
    # Set video file path
    video_path = os.path.join('videos', 'videoplayback.mp4')
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Show error if video cannot be opened
    if not cap.isOpened():
        print(f"Error: Could not open video file! ({video_path})")
        return
    
    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo Information:")
    print(f"- FPS: {fps}")
    print(f"- Total frames: {frame_count}")
    
    paused = False
    try:
        while True:
            if not paused:
                success, frame = cap.read()
                if not success:
                    print("\nReached end of video, restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Process frame
                processed_frame = process_frame(frame)
                
                # Show current frame number
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.putText(processed_frame, f'Frame: {current_frame}/{frame_count}', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow('Hand Movement Tracking System', processed_frame)
            
            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # space key
                paused = not paused
                print("Video: " + ("Paused" if paused else "Playing"))
            elif key == ord('r'):  # r key
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print("Video restarted")
    
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("\nProgram terminated.")

if __name__ == "__main__":
    main()