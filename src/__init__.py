bl_info = {
    "name": "ModernAR",
    "author": "Muammer Eren",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),  # Minimum Blender version
    "location": "View3D > Sidebar > ModernAR",
    "description": "Real-time hand gesture tracking and animation using MediaPipe.",
    "warning": "",
    "wiki_url": "",  # Optional: Add a link to documentation
    "category": "Animation",
}

# Import registration functions if Blender is running
# This prevents errors when the script is inspected externally
if "bpy" in locals():
    import importlib
    from . import cgt_registration
    importlib.reload(cgt_registration) # Ensures latest changes are picked up during development

    def register():
        cgt_registration.register()

    def unregister():
        cgt_registration.unregister()
else:
    # Define dummy functions if bpy is not available
    def register():
        print("Blender Python environment not found. Registration skipped.")
    def unregister():
        print("Blender Python environment not found. Unregistration skipped.")

# Allow running the script directly for testing outside Blender (optional)
if __name__ == "__main__":
    # This part won't run inside Blender
    print(f"Running {bl_info.get('name')} add-on script directly.")
    # You could add test code here that doesn't depend on bpy
