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
import math 
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

def GRAINCREATOR_FN_generateGrain(name, width, height):
	# LOOK AT DREAM TEXTURES TO SEE HOW ITS GENERATING IMAGES INSIDE OF BLENDER & SAVING THEM TO THE BLEND FILE 
	# lets do a single frame of grain first
	# can get width & height from render settings, not image 
	img = bpy.data.images.new(name=name, width=width, height=height)
	pixels = [1.0] * (4 * width * height)
	return img

def MATTEPAINTER_FN_contextOverride(area_to_check):
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

	def execute(self, context):
		# call func
		GRAINCREATOR_FN_generateGrain(name="myCoolGrain", width=1920, height=1080)
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
		row = layout.row()
		button_create_grain = row.operator(GRAINCREATOR_OT_generateGrain.bl_idname, text="Create Grain", icon="FILE_IMAGE")


#--------------------------------------------------------------
# Register 
#--------------------------------------------------------------

classes_interface = (GRAINCREATOR_PT_panelMain,)
classes_functionality = (GRAINCREATOR_OT_generateGrain, GRAINCREATOR_OT_createNodeGroup)

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

if __name__ == "__main__":
	register()