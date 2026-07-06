extends Control

@onready var name_label = $NameLabel
@onready var confidence_label = $ConfidenceLabel

func _ready():
	print("HUD ready")
	print("NameLabel:", name_label)
	print("ConfidenceLabel:", confidence_label)

	update_ui("Test Bird", 0.75)

func update_ui(bird_name: String, confidence: float):
	name_label.text = "Bird: " + bird_name
	confidence_label.text = "Confidence: " + str(round(confidence * 100)) + "%"

func set_hud_position(screen_pos: Vector2):
	position = screen_pos
