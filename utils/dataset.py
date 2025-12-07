import bpy
import os
import mathutils
import math
import json
import random
from bpy_extras.object_utils import world_to_camera_view

DIR = "C:/Data/cmu_mocap"
output_base = "C:/Data/mocap_dataset"
output_json = "C:/Data/mocap_dataset/87_01/annotations.json"

animations = os.listdir(DIR)

def load_animation(animation_path):
    bpy.ops.import_scene.fbx(filepath=animation_path)

def reset_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def get_3d_joints(armature):
    joints_3d = {}
    for bone in armature.pose.bones:
        world_pos = armature.matrix_world @ bone.head
        joints_3d[bone.name] = (world_pos.x, world_pos.y, world_pos.z)
    return joints_3d

def get_2d_joints(camera, joints_3d):
    scene = bpy.context.scene
    joints_2d = {}
    for name, coord in joints_3d.items():
        world_pos = mathutils.Vector(coord)
        co_2d = world_to_camera_view(scene, camera, world_pos)
        # convert normalized [0,1] to pixel coordinates
        width = scene.render.resolution_x
        height = scene.render.resolution_y
        x_px = int(co_2d.x * width)
        y_px = int((1 - co_2d.y) * height)  # Blender y=0 at bottom
        joints_2d[name] = (x_px, y_px)
    return joints_2d

def add_cameras_around_origin(num_cams=4, radius=6, heights=[2.0, 3.5]):
    cameras = []
    for i in range(num_cams):
        angle = i * (2 * math.pi / num_cams)
        for h in heights:
            cam_name = f"Cam_{i}_{h}"
            cam_data = bpy.data.cameras.new(cam_name)
            cam_obj = bpy.data.objects.new(cam_name, cam_data)
            bpy.context.collection.objects.link(cam_obj)

            cam_obj.location = mathutils.Vector((
                radius * math.cos(angle),
                radius * math.sin(angle),
                h
            ))

            direction = mathutils.Vector((0,0,1)) - cam_obj.location
            cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
            cameras.append(cam_obj)
    return cameras

def add_light(name, light_type, location, power=1000):
    light_data = bpy.data.lights.new(name, type=light_type)
    light_data.energy = power
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    return light_object

def add_floor():
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
    floor = bpy.context.active_object
    mat = bpy.data.materials.new("FloorMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1)
    floor.data.materials.append(mat)

# ------------------- Scene Setup -------------------
def render_animation(file):
    print(f"Started Rendering : {file}")
    output_folder = os.path.join(output_base, file.split(".")[0])
    frame_folder = os.path.join(output_folder, "frames")
    annotations_path = os.path.join(output_folder, "annotations.json")
    reset_scene()

    scene = bpy.context.scene

    # Load animation
    animation_path = os.path.join(DIR, file)
    load_animation(animation_path)

    armature = bpy.data.objects['CharacterArmature']
    armature.location = (0, 0, -1)
    action_name = bpy.data.actions.keys()[0]
    armature.animation_data.action = bpy.data.actions[action_name]
    action = armature.animation_data.action

    # Floor
    add_floor()

    # Lights
    add_light("KeyLight", "AREA", (4, -4, 4), power=800)
    add_light("FillLight", "AREA", (-4, -2, 3), power=300)
    add_light("RimLight", "POINT", (0, 4, 5), power=600)

    for light in [bpy.data.objects['KeyLight'],
                bpy.data.objects['FillLight'],
                bpy.data.objects['RimLight']]:
        base_energy = light.data.energy
        light.data.energy = base_energy * random.uniform(0.9, 1.1) 
        color_variation = [random.uniform(0.95, 1.05) for _ in range(3)]
        base_color = (1, 0.95, 0.9)  # slightly warm
        light.data.color = [base_color[i] * color_variation[i] for i in range(3)]

    # Cameras
    cameras = add_cameras_around_origin(num_cams=4, radius=8, heights=[2.0, 3.5])

    # ------------------- Render & Save Keypoints -------------------
    scene.frame_start = int(action.frame_range[0])
    scene.frame_end = int(action.frame_range[1])

    # Store results
    dataset = {"images": [], "annotations_2d": {}, "annotations_3d": {}}

    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)

        # 3D joints
        joints_3d = get_3d_joints(armature)
        dataset["annotations_3d"][frame] = joints_3d

        for i, cam in enumerate(cameras):
            scene.camera = cam
            img_name = f"cam{i}_frame{frame:04d}.png"
            scene.render.filepath = os.path.join(frame_folder, img_name)
            bpy.ops.render.render(write_still=True)

            # 2D joints
            joints_2d = get_2d_joints(cam, joints_3d)
            dataset["annotations_2d"][img_name] = joints_2d

        print(f"Rendered frame {frame} for {len(cameras)} cameras")

    # Save JSON
    with open(annotations_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Render complete : {file}")

for file in animations:
    render_animation(file)