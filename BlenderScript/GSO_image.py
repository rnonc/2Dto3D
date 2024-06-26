# This stub runs a python script relative to the currently open
# blend file, useful when editing scripts externally.

import bpy
import os, random, glob, sys
import numpy as np
import json

def reset_scene(resolution=128) -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.name not in {"Camera", "Area"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)
    
    context = bpy.context
    scene = context.scene
    render = scene.render
    render.engine = 'CYCLES'
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100

    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True

    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    # Set the device_type
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

def randomize_camera():
    elevation = random.uniform(0., 90.)
    azimuth = random.uniform(0., 360)
    distance = random.uniform(0.25, 0.5)
    return set_camera_location()

def set_camera_location():
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = sample_spherical(radius_min=0.4, radius_max=1, maxz=2.2, minz=-2.2)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera

def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
#         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec

def import_obj(path) -> None:
    bpy.ops.wm.obj_import(filepath=path+"/meshes/model.obj")
    model = bpy.data.objects["model"]
    rot_quat = model.location.to_track_quat('-Z', 'Y')
    model.rotation_euler = rot_quat.to_euler()
    sc = 0.2
    scale = sc*(model.dimensions.z*model.dimensions.x*model.dimensions.y)**(-1/3)
    model.scale.x = scale
    model.scale.y = scale
    model.scale.z = scale
    model.location[2] = -scale*model.dimensions.z/2
    material  = bpy.data.materials["material_0"] 
    Principled_BSDF = material.node_tree.nodes.get('Principled BSDF') 
    texImage = bpy.data.images.load(path+"/materials/textures/texture.png")
    texImage_node = material.node_tree.nodes.new('ShaderNodeTexImage')
    texImage_node.image = texImage
    material.node_tree.links.new(texImage_node.outputs[0],Principled_BSDF.inputs[0])
    
def set_light(energy=15000):
    # setup lighting
    if not "Area" in bpy.context.scene.objects:
        bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = energy
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100
    
def set_camera(lens=35,sensor_width=32):
    if not "Camera" in bpy.context.scene.objects:
        bpy.ops.object.camera_add()
    cam = bpy.context.scene.objects["Camera"]
    cam.location = (0, 0.02, 0)
    cam.data.lens = lens
    cam.data.sensor_width = sensor_width

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"


    
if __name__ == "__main__":
    path  =r"C:\Users\Admin\Documents\Dataset\GSO/"
    list_obj_path = os.listdir(path)
    obj_path = path+list_obj_path[50]
    reset_scene()
    set_light()
    set_camera()
    print(obj_path)
    import_obj(obj_path)
    randomize_camera()
    
        






