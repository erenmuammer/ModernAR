import bpy
import json
import math
import os
import sys

def find_hand_bones(armature):
    """Mixamo karakterinin el kemiklerini bul"""
    hand_bones = {
        "hand": "RightHand",
        "thumb_main": "RightHandThumb1",
        "thumb_middle": "RightHandThumb2",
        "thumb_tip": "RightHandThumb3",
        "index_main": "RightHandIndex1",
        "index_middle": "RightHandIndex2",
        "index_tip": "RightHandIndex3",
        "middle_main": "RightHandMiddle1",
        "middle_middle": "RightHandMiddle2",
        "middle_tip": "RightHandMiddle3",
        "ring_main": "RightHandRing1",
        "ring_middle": "RightHandRing2",
        "ring_tip": "RightHandRing3",
        "pinky_main": "RightHandPinky1",
        "pinky_middle": "RightHandPinky2",
        "pinky_tip": "RightHandPinky3"
    }
    
    found_bones = {}
    for our_name, mixamo_name in hand_bones.items():
        if mixamo_name in armature.pose.bones:
            found_bones[our_name] = armature.pose.bones[mixamo_name]
    return found_bones

def update_hand_animation(json_path):
    """JSON dosyasından el hareketlerini güncelle ve render al"""
    try:
        # JSON dosyasını oku
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Aktif armatürü al
        armature = bpy.context.active_object
        if armature.type != 'ARMATURE':
            raise Exception("Karakter armatürü seçili değil!")
        
        # El kemiklerini bul
        hand_bones = find_hand_bones(armature)
        
        # Son frame'i al
        if data["frames"]:
            last_frame = data["frames"][-1]
            landmarks = last_frame["landmarks"]
            
            if landmarks:
                # El bileği pozisyonu ve rotasyonu
                wrist = landmarks[0]
                if 'hand' in hand_bones:
                    bone = hand_bones['hand']
                    bone.rotation_euler.x = (wrist['x'] - 0.5) * math.pi
                    bone.rotation_euler.y = (wrist['y'] - 0.5) * math.pi
                    bone.rotation_euler.z = (wrist['z'] - 0.5) * math.pi
                
                # Parmak kemikleri için
                finger_mapping = {
                    "thumb": [1, 2, 3],
                    "index": [5, 6, 7],
                    "middle": [9, 10, 11],
                    "ring": [13, 14, 15],
                    "pinky": [17, 18, 19]
                }
                
                for finger, indices in finger_mapping.items():
                    for i, bone_type in enumerate(["main", "middle", "tip"]):
                        bone_name = f"{finger}_{bone_type}"
                        if bone_name in hand_bones and indices[i] < len(landmarks):
                            bone = hand_bones[bone_name]
                            landmark = landmarks[indices[i]]
                            bone.rotation_euler.x = (landmark['x'] - 0.5) * math.pi
                            bone.rotation_euler.y = (landmark['y'] - 0.5) * math.pi
                            bone.rotation_euler.z = (landmark['z'] - 0.5) * math.pi
        
        # Render ayarları
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = "render/frame.png"
        
        # Render al
        bpy.ops.render.render(write_still=True)
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")

if __name__ == "__main__":
    # JSON dosya yolunu komut satırı argümanlarından al
    json_path = sys.argv[-1]
    update_hand_animation(json_path)