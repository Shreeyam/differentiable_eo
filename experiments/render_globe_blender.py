"""
Render the Earth globe middle layer in Blender (Cycles).

Produces an opaque textured Earth sphere with transparent background (RGBA PNG).
Compositing with vector back/front orbital layers is done in gen_globe_figures.py.

Usage:
    /Applications/Blender.app/Contents/MacOS/Blender --background --python experiments/render_globe_blender.py
"""
import bpy, math, os

ELEV, AZIM = 25, 45
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXTURE_PATH = os.path.join(SCRIPT_DIR, '..', 'earth_nasa.jpg')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '..', 'paper', 'figures', 'globe_earth_layer.png')

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 128
scene.render.resolution_x = 2048
scene.render.resolution_y = 2048
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'

# Camera — ortho, unit sphere radius=1, scale=2 so globe just touches edges
cam = bpy.data.cameras.new('Cam')
cam.type = 'ORTHO'
cam.ortho_scale = 2.0
cam.clip_start = 0.01
cam.clip_end = 100
cam_obj = bpy.data.objects.new('Cam', cam)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

el_r, az_r = math.radians(ELEV), math.radians(AZIM)
d = 10
cam_obj.location = (d * math.cos(el_r) * math.cos(az_r),
                     d * math.cos(el_r) * math.sin(az_r),
                     d * math.sin(el_r))

track = cam_obj.constraints.new(type='TRACK_TO')
empty = bpy.data.objects.new('Target', None)
scene.collection.objects.link(empty)
track.target = empty
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

bpy.context.evaluated_depsgraph_get().update()

# Unit sphere with Earth texture
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=128, ring_count=64)
sphere = bpy.context.active_object
bpy.ops.object.shade_smooth()

mat = bpy.data.materials.new('Earth')
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

tex_coord = nodes.new('ShaderNodeTexCoord')
tex_img = nodes.new('ShaderNodeTexImage')
tex_img.image = bpy.data.images.load(TEXTURE_PATH)
links.new(tex_coord.outputs['UV'], tex_img.inputs['Vector'])

emission = nodes.new('ShaderNodeEmission')
emission.inputs['Strength'].default_value = 1.0
links.new(tex_img.outputs['Color'], emission.inputs['Color'])

output = nodes.new('ShaderNodeOutputMaterial')
links.new(emission.outputs['Emission'], output.inputs['Surface'])

sphere.data.materials.append(mat)

# No world lighting
world = bpy.data.worlds.new('W')
scene.world = world
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.0

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
scene.render.filepath = OUTPUT_PATH
bpy.ops.render.render(write_still=True)
print(f"Saved: {OUTPUT_PATH}")
