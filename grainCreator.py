#--------------------------------------------------------------
# Meta Dictionary
#--------------------------------------------------------------

bl_info = {
	"name" : "GrainCreator",
	"author" : "SceneFiller",
	"version" : (1, 0, 0),
	"blender" : (4, 0, 2),
	"location" : "View3d > Tool", # change me
	"warning" : "",
	"wiki_url" : "",
	"category" : "3D View", # change me 
}

#--------------------------------------------------------------
# Import
#--------------------------------------------------------------

import os
import bpy
import bpy_extras
import numpy as np
from bpy.props import PointerProperty, BoolProperty
from bpy_extras.image_utils import load_image
from pathlib import Path
from bpy_extras.io_utils import ImportHelper
import time, sys

#--------------------------------------------------------------
# Miscellaneous Functions
#--------------------------------------------------------------

def _convert_pixel_buffer_to_matrix(buffer, width, height, channels):
	# Converts a 1-D pixel buffer into an xy grid with n Colour channels
	buffer = buffer.reshape(height, width, channels)
	return buffer

def _convert_matrix_to_pixel_buffer(buffer):
	# Converts back to 1-D pixel buffer
	buffer = buffer.flatten()
	return buffer		

''
def _filter_gaussian(l=3, sig=1.0):
	ax = np.linspace(-(l-1) / 2.0, (l-1) /2.0, l)
	gaussian = np.exp(-0.5 * np.square(ax) / np.square(sig))
	kernel = np.outer(gaussian, gaussian)
	return kernel / np.sum(kernel)

def GRAINCREATOR_FN_generateGrain(name):
	w = bpy.data.scenes[0].render.resolution_x
	h = bpy.data.scenes[0].render.resolution_y	
	buffer_size = (4 * w * h)

	clip_min = .4
	clip_max = .7

	# KEEP ME
	img = bpy.data.images.new(name=name, width=w, height=h)
	
	pixels_to_paint = np.ones(buffer_size, dtype=np.float32)	
	pixels_to_paint = _convert_pixel_buffer_to_matrix(pixels_to_paint, w, h, 4)	

	# GENERATE NOISE HERE
	# ------------------------

	color_grain = False

	# Editors can calibrate grain size, structure, and texture to achieve the desired effect.
	r = np.random.rand(*pixels_to_paint.shape)

	if color_grain:
		pixels_to_paint[:, :, 0:3] = r[:, :, 0:3]
	else:
		pixels_to_paint[:, :, 0:3] = r[:, :, 0:1]	

	pixels_to_paint = np.clip(pixels_to_paint, clip_min, clip_max)

	# Apply Gaussian Blur
	kernel = _filter_gaussian()
	blur_result = np.convolve(pixels_to_paint.flatten(), kernel.flatten())

	pixels = blur_result[:buffer_size].reshape(pixels_to_paint.shape)
	pixelsf32 = pixels.astype(np.float32)

	# Fix Alpha to 1.0
	pixelsf32[:, :, 3] = 1.0

	# ------------------------
	output = _convert_matrix_to_pixel_buffer(pixelsf32)
	img.pixels.foreach_set(output)
	img.pack()
	img.update()	

	# Show generated grain in Image Editor 
	for area in bpy.context.screen.areas:
		if area.type == 'IMAGE_EDITOR':
			area.spaces.active.image = img

	return img

def GRAINCREATOR_FN_addNodeGroupToCompositor():
	# just create a node group & add it to compositor, user can plug the shit in manually 
	return 

def GRAINCREATOR_FN_contextOverride(area_to_check):
	return [area for area in bpy.context.screen.areas if area.type == area_to_check][0]

#--------------------------------------------------------------
# Operators
#--------------------------------------------------------------	

class GRAINCREATOR_OT_generateGrain(bpy.types.Operator):	
	bl_idname = "graincreator.generate_grain"
	bl_label = "Generates custom film grain image or sequence."
	bl_options = {"REGISTER", "UNDO"}
	bl_description = "Generates custom film grain image or sequence."

	start: bpy.props.IntProperty(name='start', default=1)
	end: bpy.props.IntProperty(name='end', default=1)
	sequence: bpy.props.BoolProperty(name='sequence', default=True)

	def execute(self, context):
		# Assert valid frame range.
		if self.end < self.start:
			self.report({"WARNING"}, "Invalid frame range.")	
			return {'CANCELLED'}

		sequence_length = (self.end - self.start) + 1 if self.end > self.start else 1

		seq = []

		for i in range(sequence_length):
			grain = GRAINCREATOR_FN_generateGrain(name=f"grain_{i+1}")
			if sequence_length > 1:
				seq.append(grain)

		scene = bpy.context.scene 
		compositor_node_tree = scene.node_tree
		image_node = compositor_node_tree.nodes.new(type="CompositorNodeImage")

		if sequence_length == 1:
			image_node.image = grain
		if sequence_length > 1: 		
			image_node.image = seq[0]		
			# when changing to SEQUENCE, blender is unable to "load" the image	
			#image_node.image.source = 'SEQUENCE'
			#image_node.frame_duration = len(seq)
			
		return {'FINISHED'}

class GRAINCREATOR_OT_createNodeGroup(bpy.types.Operator):
	bl_idname = "graincreator.create_node_group"
	bl_label = "Film Grain node."
	bl_options = {"REGISTER", "UNDO"}
	bl_description = "Film Grain node."

	def execute(self, context):
		return{'FINISHED'}

############# TEMP
class GRAINCREATOR_OT_clearUnused(bpy.types.Operator):
	# Purges unused Data Blocks.
	bl_idname = "graincreator.clear_unused"
	bl_label = "Clear Unused"
	bl_description = "Removes unlinked data from the Blend File. WARNING: This process cannot be undone"
	bl_options = {"REGISTER"}

	def execute(self, context):
		bpy.ops.outliner.orphans_purge('INVOKE_DEFAULT' if True else 'EXEC_DEFAULT', num_deleted=0, do_local_ids=True, do_linked_ids=False, do_recursive=True)
		return {'FINISHED'}

#--------------------------------------------------------------
# Interface
#--------------------------------------------------------------

class GRAINCREATOR_PT_panelMain(bpy.types.Panel):
	bl_label = "Grain Creator"
	bl_idname = "GRAINCREATOR_PT_panelMain"
	bl_space_type = 'NODE_EDITOR'
	bl_region_type = 'UI'
	bl_category = 'Grain Creator'

	@classmethod 
	def poll(cls, context):
		snode = context.space_data
		return snode.tree_type == 'CompositorNodeTree'

	def draw(self, context):
		layout = self.layout		
		view = context.space_data
		scene = context.scene

		# Grain Settings

		# Start & End Frames
		row = layout.row()
		row.prop(context.scene, 'GRAINCREATOR_VAR_start', text='Start')
		row.prop(context.scene, 'GRAINCREATOR_VAR_end', text='End')

		# Create Grain
		row = layout.row()
		button_create_grain = row.operator(GRAINCREATOR_OT_generateGrain.bl_idname, text="Create Grain", icon="FILE_IMAGE")

		# Save Grain 
		row = layout.row()
		button_export_grain = row.operator(GRAINCREATOR_OT_generateGrain.bl_idname, text="Export Grain", icon_value=727)

		

		button_create_grain.start = context.scene.GRAINCREATOR_VAR_start
		button_create_grain.end = context.scene.GRAINCREATOR_VAR_end

		row = layout.row()
		button_purge = row.operator(GRAINCREATOR_OT_clearUnused.bl_idname, text="purge(TEMP)", icon_value=727)

		



#--------------------------------------------------------------
# Register 
#--------------------------------------------------------------

classes_interface = (GRAINCREATOR_PT_panelMain,)
classes_functionality = (GRAINCREATOR_OT_generateGrain, GRAINCREATOR_OT_createNodeGroup, GRAINCREATOR_OT_clearUnused)

bpy.types.Scene.GRAINCREATOR_VAR_start = bpy.props.IntProperty(name='GRAINCREATOR_VAR_start', default=1, soft_min=0, description='Frame start for Grain generation.')
bpy.types.Scene.GRAINCREATOR_VAR_end = bpy.props.IntProperty(name='GRAINCREATOR_VAR_end', default=1, soft_min=0, description='Frame end for Grain generation.')

def register():

	# Register Classes
	for c in classes_interface:
		bpy.utils.register_class(c)
	for c in classes_functionality:
		bpy.utils.register_class(c)	
			
def unregister():

	# Unregister
	for c in reversed(classes_interface):
		bpy.utils.unregister_class(c)
	for c in reversed(classes_functionality):
		bpy.utils.unregister_class(c)

	del bpy.types.Scene.GRAINCREATOR_VAR_start
	del bpy.types.Scene.GRAINCREATOR_VAR_end

if __name__ == "__main__":
	register()