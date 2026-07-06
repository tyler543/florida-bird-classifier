extends Node3D

@onready var hud = $HUDLayer/HUD
@onready var box = $HUDLayer/Box
@onready var camera = $Camera3D
@onready var bird = $Bird

func _ready():
	await get_tree().process_frame
	
	hud.update_ui("Loading...", 0.0)

	await get_tree().create_timer(2.0).timeout

	hud.update_ui("Cardinal", 0.87)

func _process(delta):
	if bird == null:
		return

	var world_pos = bird.global_transform.origin
	var screen_pos = camera.unproject_position(world_pos)

	# Move HUD
	hud.set_hud_position(screen_pos)

	# Create a simple box around the bird (screen space)
	var box_size = Vector2(120, 80)  # adjust as needed

	var rect = Rect2(
		screen_pos - box_size * 0.5,
		box_size
	)

	box.set_box(rect)
