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
import cv2
import numpy as np
from bpy.props import PointerProperty, BoolProperty
import math 
import random
from mathutils import Vector
import mathutils
from bpy_extras.image_utils import load_image
from pathlib import Path
import shutil
from bpy_extras import view3d_utils
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

def GRAINCREATOR_FN_generateGrain(name):
	w = bpy.data.scenes[0].render.resolution_x
	h = bpy.data.scenes[0].render.resolution_y	

	clip_min = .4
	clip_max = .7

	img = bpy.data.images.new(name=name, width=w, height=h)
	pixels_to_paint = np.ones(4 * w * h, dtype=np.float32)	

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

	# ------------------------
	pixels_to_paint = _convert_matrix_to_pixel_buffer(pixels_to_paint)
	img.pixels.foreach_set(pixels_to_paint)
	img.update()	

	return img

def GRAINCREATOR_FN_addNodeGroupToCompositor():
	# just create a node group & add it to compositor, user can plug the shit in manually 
	return 

def GRAINCREATOR_FN_contextOverride(area_to_check):
	return [area for area in bpy.context.screen.areas if area.type == area_to_check][0]

#--------------------------------------------------------------
# Layer Creation
#--------------------------------------------------------------	

'''
1. create image/sequence and load it into a node 
2. save the image/sequence inside the blend file (maybe... yikes)
3. alternatively, give user the option to export the grain into a video
4. and let them sell it :) 
5. group input -> image node -> mix node -> exposure node -> group output
                  value node -> math (mult) -----^
6. DONESKIES 
'''	

class GRAINCREATOR_OT_generateGrain(bpy.types.Operator):	
	bl_idname = "graincreator.generate_grain"
	bl_label = "Generates custom film grain image or sequence."
	bl_options = {"REGISTER", "UNDO"}
	bl_description = "Generates custom film grain image or sequence."

	start: bpy.props.IntProperty(name='start', default=1)
	end: bpy.props.IntProperty(name='end', default=1)

	def execute(self, context):
		# Assert valid frame range.
		if self.end < self.start:
			self.report({"WARNING"}, "Invalid frame range.")	
			return {'CANCELLED'}

		sequence_length = (self.end - self.start) + 1 if self.end > self.start else 1

		for i in range(sequence_length):
			GRAINCREATOR_FN_generateGrain(name=f"grain_{i+1}")
		return {'FINISHED'}

class GRAINCREATOR_OT_createNodeGroup(bpy.types.Operator):
	bl_idname = "graincreator.create_node_group"
	bl_label = "Film Grain node."
	bl_options = {"REGISTER", "UNDO"}
	bl_description = "Film Grain node."

	def execute(self, context):
		return{'FINISHED'}

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

		# Create Grain
		row = layout.row()
		button_create_grain = row.operator(GRAINCREATOR_OT_generateGrain.bl_idname, text="Create Grain", icon="FILE_IMAGE")

		# Save Grain 
		row = layout.row()
		button_export_grain = row.operator(GRAINCREATOR_OT_generateGrain.bl_idname, text="Export Grain", icon_value=727)

		# Start & End Frames
		row = layout.row()
		row.prop(context.scene, 'GRAINCREATOR_VAR_start', text='Start')
		row.prop(context.scene, 'GRAINCREATOR_VAR_end', text='End')

		button_create_grain.start = context.scene.GRAINCREATOR_VAR_start
		button_create_grain.end = context.scene.GRAINCREATOR_VAR_end

		



#--------------------------------------------------------------
# Register 
#--------------------------------------------------------------

classes_interface = (GRAINCREATOR_PT_panelMain,)
classes_functionality = (GRAINCREATOR_OT_generateGrain, GRAINCREATOR_OT_createNodeGroup)

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