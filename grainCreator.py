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
from functools import partial 

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
def _filter_gaussian(k=3, sig=1.0):
	ax = np.linspace(-(k-1) / 2.0, (k-1) /2.0, k)
	gaussian = np.exp(-0.5 * np.square(ax) / np.square(sig))
	kernel = np.outer(gaussian, gaussian)
	return kernel / np.sum(kernel)

def GRAINCREATOR_FN_generateGrain(name, clip_min=.4, clip_max=.7, k=3, sigma=1.0, oversampling=False, monochromatic=True):	
	w = bpy.data.scenes[0].render.resolution_x if not oversampling else (bpy.data.scenes[0].render.resolution_x * 2)
	h = bpy.data.scenes[0].render.resolution_y if not oversampling else (bpy.data.scenes[0].render.resolution_y * 2)
	buffer_size = (4 * w * h)

	# Create Blank Image
	grain = bpy.data.images.new(name=name, width=w, height=h)
	
	# Convert Pixel Buffer to Matrix
	pixels_to_paint = np.ones(buffer_size, dtype=np.float32)	
	pixels_to_paint = _convert_pixel_buffer_to_matrix(pixels_to_paint, w, h, 4)	

	# Generate Grain
	r = np.random.rand(*pixels_to_paint.shape)

	if monochromatic:
		pixels_to_paint[:, :, 0:3] = r[:, :, 0:1]	
	else:
		pixels_to_paint[:, :, 0:3] = r[:, :, 0:3]		

	# Clip Values
	pixels_to_paint = np.clip(pixels_to_paint, clip_min, clip_max)

	# Apply Gaussian Blur
	kernel = _filter_gaussian(k=k, sig=sigma)
	blur_result = np.convolve(pixels_to_paint.flatten(), kernel.flatten())
	pixels = blur_result[:buffer_size].reshape(pixels_to_paint.shape)
	pixels_f32 = pixels.astype(np.float32)

	# Fix Alpha to 1.0
	pixels_f32[:, :, 3] = 1.0
	
	# Propagate Grain to Empty Image
	output = _convert_matrix_to_pixel_buffer(pixels_f32)
	grain.pixels.foreach_set(output)	
	grain.update()	

	# Show Result in Image Editor 
	for area in bpy.context.screen.areas:
		if area.type == 'IMAGE_EDITOR':
			area.spaces.active.image = grain

	return grain

def GRAINCREATOR_FN_exportFrame(image, idx, dir):
	name = idx
	if idx < 10:
		name = f'000{idx}'
	if idx >= 10 and idx < 100:
		name = f'00{idx}'
	if idx >= 100 and idx < 1000:
		name = f'0{idx}'
	image.filepath_raw = f'{dir}{name}.png'
	image.save()

def GRAINCREATOR_FN_addNodeGroupToCompositor():
	# just create a node group & add it to compositor, user can plug the shit in manually 
	return 

def GRAINCREATOR_FN_contextOverride(area_to_check):
	return [area for area in bpy.context.screen.areas if area.type == area_to_check][0]

def GRAINCREATOR_FN_refreshUI():
	for region in bpy.context.area.regions:
		if region.type == 'UI':
			region.tag_redraw()
	return None 
	'''
	for area in bpy.context.screen.areas:
		for region in area.regions:
			if region.type == "UI":
				region.tag_redraw()
	'''

#--------------------------------------------------------------
# Operators
#--------------------------------------------------------------	

class GRAINCREATOR_OT_Timer(bpy.types.Operator):
	bl_idname = "graincreator.timer"
	bl_label = "Timer Object"

	_timer = None

	def modal(self, context, event):
		GRAINCREATOR_FN_refreshUI()
		#[a.tag_redraw() for a in context.screen.areas]
		if self._timer.time_duration > 3:
			context.window_manager.progress = 1
			return {'FINISHED'}
		context.window_manager.progress = self._timer.time_duration / 3
		return {'PASS_THROUGH'}

	def execute(self, context):
		wm = context.window_manager
		self._timer = wm.event_timer_add(0.1, window=context.window)
		wm.modal_handler_add(self)
		return {'RUNNING_MODAL'}

class GRAINCREATOR_OT_generateGrain(bpy.types.Operator):	
	bl_idname = "graincreator.generate_grain"
	bl_label = "Generates custom film grain image or sequence."
	bl_options = {"REGISTER", "UNDO"}
	bl_description = "Generates custom film grain image or sequence."

	clip_min: bpy.props.FloatProperty(name='clip_min', default=.4)
	clip_max: bpy.props.FloatProperty(name='clip_max', default=.7)
	kernel_size: bpy.props.IntProperty(name='kernel_size', default=3)
	sigma: bpy.props.FloatProperty(name='sigma', default=1.0)
	oversampling: bpy.props.BoolProperty(name='oversampling', default=False)
	monochromatic: bpy.props.BoolProperty(name='monochromatic', default=False)

	def execute(self, context):
		# Assert valid Clipping
		if self.clip_max < self.clip_min:
			self.report({"WARNING"}, "Invalid clip range.")
			return{'CANCELLED'}

		# Generate single frame for previewing.
		grain = GRAINCREATOR_FN_generateGrain(
			name=f"grainSingleFrame", 
			clip_min=self.clip_min, 
			clip_max=self.clip_max, 
			k=self.kernel_size, 
			sigma=self.sigma,
			oversampling=self.oversampling,
			monochromatic=self.monochromatic)			
		return {'FINISHED'}

class GRAINCREATOR_OT_exportGrainFrames(bpy.types.Operator):	
	bl_idname = "graincreator.export_grain_frames"
	bl_label = "Exports grain sequence to output folder."
	bl_options = {"REGISTER", "UNDO"}
	bl_description = "Exports grain sequence to output folder."

	clip_min: bpy.props.FloatProperty(name='clip_min', default=.4)
	clip_max: bpy.props.FloatProperty(name='clip_max', default=.7)
	kernel_size: bpy.props.IntProperty(name='kernel_size', default=3)
	sigma: bpy.props.FloatProperty(name='sigma', default=1.0)
	oversampling: bpy.props.BoolProperty(name='oversampling', default=False)
	monochromatic: bpy.props.BoolProperty(name='monochromatic', default=False)
	frames: bpy.props.IntProperty(name='frames', default=1)

	def execute(self, context):
		# Assert valid Clipping
		if self.clip_max < self.clip_min:
			self.report({"WARNING"}, "Invalid clip range.")
			return{'CANCELLED'}

		# Export grain frames.	
		bpy.ops.wm.console_toggle()
		print('Writing Frames...')
		for i in range(self.frames):
			print(f'{i}/{self.frames}')
			grain = GRAINCREATOR_FN_generateGrain(
				name=f"{i+1}",
				clip_min=self.clip_min, 
				clip_max=self.clip_max, 
				k=self.kernel_size, 
				sigma=self.sigma,
				oversampling=self.oversampling,
				monochromatic=self.monochromatic)		
			GRAINCREATOR_FN_exportFrame(grain, i+1, "D:/Documents/Scenefiller/grainCreator/output/")
		print('Finishing up...')
		bpy.ops.wm.console_toggle()
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

		# Output 
		row = layout.row()
		row.label(text='Grain Generation', icon='REMOVE')	

		# Grain Settings
		row = layout.row()
		row.prop(context.scene, 'GRAINCREATOR_VAR_clip_min', text='Clip Min')
		row.prop(context.scene, 'GRAINCREATOR_VAR_clip_max', text='Clip Max')
		row = layout.row()
		row.prop(context.scene, 'GRAINCREATOR_VAR_kernel', text='Kernel')
		row.prop(context.scene, 'GRAINCREATOR_VAR_sigma', text='Sigma')
		row = layout.row()
		row.prop(context.scene, 'GRAINCREATOR_VAR_oversampling', text='Oversampling')
		row = layout.row()
		row.prop(context.scene, 'GRAINCREATOR_VAR_monochromatic', text='Monochromatic')
		
		# Create Grain
		row = layout.row()
		button_create_grain = row.operator(GRAINCREATOR_OT_generateGrain.bl_idname, text="Create Grain", icon="FILE_IMAGE")
	
		row = layout.row()
		button_purge = row.operator(GRAINCREATOR_OT_clearUnused.bl_idname, text="purge(TEMP)", icon_value=727)

		########################
		# need to add grain preview image here.
		########################

		# Output 
		row = layout.row()
		row.label(text='Output', icon='REMOVE')	

		# Output Directory
		row = layout.row()
		row.label(text='Output Folder: ')
		row = layout.row()
		row.prop(scene, 'MATTECREATOR_VAR_outputDir')

		# Frame Count
		row = layout.row()
		row.prop(scene, 'GRAINCREATOR_VAR_frames', text='Frames')

		# Save Grain 
		row = layout.row()
		button_export_grain = row.operator(GRAINCREATOR_OT_exportGrainFrames.bl_idname, text='Export Frames', icon_value=727)	

		# Assign Variables
		button_create_grain.clip_min = context.scene.GRAINCREATOR_VAR_clip_min
		button_create_grain.clip_max = context.scene.GRAINCREATOR_VAR_clip_max
		button_create_grain.kernel_size = context.scene.GRAINCREATOR_VAR_kernel
		button_create_grain.sigma = context.scene.GRAINCREATOR_VAR_sigma		
		button_create_grain.oversampling = context.scene.GRAINCREATOR_VAR_oversampling
		button_create_grain.monochromatic = context.scene.GRAINCREATOR_VAR_monochromatic

		button_export_grain.clip_min = context.scene.GRAINCREATOR_VAR_clip_min
		button_export_grain.clip_max = context.scene.GRAINCREATOR_VAR_clip_max
		button_export_grain.kernel_size = context.scene.GRAINCREATOR_VAR_kernel
		button_export_grain.sigma = context.scene.GRAINCREATOR_VAR_sigma		
		button_export_grain.oversampling = context.scene.GRAINCREATOR_VAR_oversampling
		button_export_grain.monochromatic = context.scene.GRAINCREATOR_VAR_monochromatic
		button_export_grain.frames = context.scene.GRAINCREATOR_VAR_frames


#--------------------------------------------------------------
# Register 
#--------------------------------------------------------------

classes_interface = (GRAINCREATOR_PT_panelMain,)
classes_functionality = (GRAINCREATOR_OT_generateGrain, GRAINCREATOR_OT_exportGrainFrames, GRAINCREATOR_OT_createNodeGroup, GRAINCREATOR_OT_clearUnused)

bpy.types.Scene.GRAINCREATOR_VAR_clip_min = bpy.props.FloatProperty(name='GRAINCREATOR_VAR_clip_min', default=.4, soft_min=0.0, soft_max=1.0, description='Squash Black Values in Generated Grain.')
bpy.types.Scene.GRAINCREATOR_VAR_clip_max = bpy.props.FloatProperty(name='GRAINCREATOR_VAR_clip_max', default=.7, soft_min=0.0, soft_max=1.0, description='Squash White Values in Generated Grain.')
bpy.types.Scene.GRAINCREATOR_VAR_kernel = bpy.props.IntProperty(name='GRAINCREATOR_VAR_kernel', default=3, soft_min=1, soft_max=16, description='Set Kernel Size for Gaussian Blur.')
bpy.types.Scene.GRAINCREATOR_VAR_sigma = bpy.props.FloatProperty(name='GRAINCREATOR_VAR_sigma', default=1.0, soft_min=0.0, soft_max=5.0, description='Set Sigma for Gaussian Blur.')
bpy.types.Scene.GRAINCREATOR_VAR_oversampling = bpy.props.BoolProperty(name='GRAINCREATOR_VAR_oversampling', default=False)
bpy.types.Scene.GRAINCREATOR_VAR_monochromatic = bpy.props.BoolProperty(name='GRAINCREATOR_VAR_monochromatic', default=True)
bpy.types.Scene.GRAINCREATOR_VAR_frames = bpy.props.IntProperty(name='GRAINCREATOR_VAR_frames', default=1, soft_min=1, description='Number of frames to export.')
bpy.types.Scene.GRAINCREATOR_VAR_outputDir = bpy.props.StringProperty(name='', default='', subtype='DIR_PATH')

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

	del bpy.types.Scene.GRAINCREATOR_VAR_clip_min
	del bpy.types.Scene.GRAINCREATOR_VAR_clip_max
	del bpy.types.Scene.GRAINCREATOR_VAR_kernel
	del bpy.types.Scene.GRAINCREATOR_VAR_sigma	
	del bpy.types.Scene.GRAINCREATOR_VAR_oversampling 
	del bpy.types.Scene.GRAINCREATOR_VAR_monochromatic	
	del bpy.types.Scene.GRAINCREATOR_VAR_frames
	del bpy.types.Scene.GRAINCREATOR_VAR_outputDir

if __name__ == "__main__":
	register()