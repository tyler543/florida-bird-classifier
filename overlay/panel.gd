extends Panel

@onready var label: Label = $Label

func _ready():
	apply_base_style()

# Call this to update the UI dynamically
func update_ui(bird_name: String, confidence: float):
	label.text = bird_name + " (" + str(round(confidence * 100)) + "%)"
	update_border_color(confidence)

# Apply the base HUD-style look
func apply_base_style():
	var style = StyleBoxFlat.new()

	# Background (semi-transparent dark)
	style.bg_color = Color(0, 0, 0, 0.5)

	# Border
	style.border_width_left = 2
	style.border_width_right = 2
	style.border_width_top = 2
	style.border_width_bottom = 2

	# Default border color (will be overridden later)
	style.border_color = Color(0.2, 0.8, 1.0)

	# Padding
	style.content_margin_left = 10
	style.content_margin_right = 10
	style.content_margin_top = 6
	style.content_margin_bottom = 6

	# Rounded corners
	style.corner_radius_top_left = 6
	style.corner_radius_top_right = 6
	style.corner_radius_bottom_left = 6
	style.corner_radius_bottom_right = 6

	add_theme_stylebox_override("panel", style)

	# Text styling (white, readable)
	label.add_theme_color_override("font_color", Color(1, 1, 1))


# Dynamically change border color based on confidence
func update_border_color(confidence: float):
	var style: StyleBoxFlat = get_theme_stylebox("panel")

	if confidence >= 0.8:
		style.border_color = Color(0.2, 1.0, 0.3) # green (high confidence)
	elif confidence >= 0.5:
		style.border_color = Color(1.0, 0.8, 0.2) # yellow (medium)
	else:
		style.border_color = Color(1.0, 0.3, 0.3) # red (low)

	add_theme_stylebox_override("panel", style)
