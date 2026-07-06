# cd "OneDrive\Desktop" && python send_test.py

extends Control

# Creates and stores tmeporary variable data 
var detections = []
var hud_color := Color.RED
var hud_fields := [
	"Common Name",
	"Scientific Name",
	"Confidence",
    "Conservation Status"
]
var hud_layout := "layout1"

# --------------------------------------------------------------
# RECEIVE HUD SETTINGS 
# Python sends new HUD settings
# --------------------------------------------------------------
func update_hud_config(config):

	# Converts HTML color string into Godot Color
	if config.has("hud_color"):
		hud_color = Color.html(config["hud_color"])

	# Updates what text fields should appear
	if config.has("hud_fields"):
		hud_fields = config["hud_fields"]
	
	# Changes visual layout
	if config.has("hud_layout"):
		hud_layout = config["hud_layout"]

	print("HUD CONFIG RECEIVED")
	print("Color:", hud_color)
	print("Fields:", hud_fields)
	print("Layout:", hud_layout)

	# Redraw the screen when it updates the data
	queue_redraw()

# --------------------------------------------------------------
# RECEIVE DETECTIONS 
# replace old detections with new one and redraw UI
# --------------------------------------------------------------
func update_detections(new_detections):
	detections = new_detections
	queue_redraw()

# --------------------------------------------------------------
# DRAW
# Called whenever the UI is drawn
# --------------------------------------------------------------
func _draw():

	# Uses default UI
	var font = ThemeDB.fallback_font

	# Loop through detections (each detection == one bounding box)
	for detection in detections:

		# Extract box values (0 is used as default value in case value is missing)
		var x = detection.get("x", 0)
		var y = detection.get("y", 0)
		var w = detection.get("w", 0)
		var h = detection.get("h", 0)

		# Draws transparent box with 20% opacity fill
		var fill_color = Color(
			hud_color.r,
			hud_color.g,
			hud_color.b,
			0.2
		)

		# Fill rectangle 
		draw_rect(
			Rect2(x, y, w, h),
			fill_color,
			true
		)

		# Draw rectangle outline
		draw_rect(
			Rect2(x, y, w, h),
			hud_color,
			false,
			1.0
		)

		# Checks which layout to draw: layout1, layout2, layout3 based on JSON
		match hud_layout:
			"layout1":
				_draw_layout1(x, y, w, h, detection, font)

			"layout2":
				_draw_layout2(x, y, w, h, detection, font)

			"layout3":
				_draw_layout3(x, y, w, h, detection, font)

			_:
				_draw_layout1(x, y, w, h, detection, font)

# --------------------------------------------------------------
# Get Field Helper function
# --------------------------------------------------------------
func get_field(index: int) -> String:
	
	# Checks if index exits and value is not null, return as string
	if index < hud_fields.size() and hud_fields[index] != null:
		return str(hud_fields[index])
	return ""

# --------------------------------------------------------------
# Boundary Helper function
# Keeps text inside bounding box
# --------------------------------------------------------------
func clamp_to_box(pos: Vector2, box: Rect2) -> Vector2:
	return Vector2(
		clamp(pos.x, box.position.x + 5, box.position.x + box.size.x - 5),
		clamp(pos.y, box.position.y + 5, box.position.y + box.size.y - 5)
	)

# --------------------------------------------------------------
# LAYOUT 1 and Helper Functions
# --------------------------------------------------------------
func _draw_layout1(x, y, w, h, detection, font):

	# Creates rectangle
	var box = Rect2(x, y, w, h)
	
	# Set font size
	var font_size = 18
	
	# Choose what fields to show
	var fields = [
		get_field(0),
		get_field(1),
		get_field(2),
		get_field(3)
	]

	# Draw text slightly below top of box
	var y_offset = box.position.y + 5

	# Loop throught fields and skip empty fields
	for field in fields:
		if field == "":
			continue

		# Convert field name to real data (_resolve_field translates)
		var text = _resolve_field(field, detection)
		
		# Place text slightly right of box edge and vertically stacked
		var pos = Vector2(box.position.x + 5, y_offset + font_size)

		# Makes sure text doesn't go outside bouding box
		pos = clamp_to_box(pos, box)

		# Draws background box with text on top
		draw_label_with_bg(
			font,
			text,
			pos,
			font_size,
			hud_color
		)

		# Move down for next label
		y_offset += font_size + 6

func draw_label_with_bg(font, text: String, pos: Vector2, font_size: int, text_color: Color):

	# Measures how many pixels wide the text is: "Blue Jay: -> 80 pixels
	var text_size = font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size)

	# Adds space around text: 80 pixles + 6 = 86 pixles (dynamic box)
	var padding = 6

	# Draws background rectangle with dark transparent background
	draw_rect(
		Rect2(
			pos.x - padding,
			pos.y - font_size,
			text_size.x + padding * 2,
			font_size + padding
		),
		Color(0, 0, 0, 0.6),  # dark transparent background
		true
	)

	# Draws text on top
	draw_string(
		font,
		pos,
		text,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		font_size,
		text_color
	)

func _resolve_field(field: String, detection: Dictionary) -> String:

	# Converts each "field name" -> actual detection
	# if detection is "commonLable": "Blue Jay" -> "Blue Jay"
	# if detection is "scientificLabel": "Cyanocitta Cristata" -> "Cyanocitta Cristata"
	# if detection is "confidence":  0.87 -> 0.872*100= round(87.2) = 87 + "%" = 87%
	# if field is null, return empty string

	match field:
		"Common Name":
			return str(detection.get("common_name", "Unknown"))

		"Scientific Name":
			return str(detection.get("scientific_name", "Unknown"))

		"Confidence":
			return "Conf: " + str(round(float(detection.get("confidence", 0.0)) * 100.0)) + "%"
		
		"Conservation Status":
			return str(detection.get("conservation_status", "Unknown"))
		_:
			return ""
		
		
# --------------------------------------------------------------
# LAYOUT 2 and Helper Functions
# --------------------------------------------------------------
func _draw_layout2(x, y, w, _h, detection, font):

	# Creates basic settings for banner box:
	# height, corner roundness, title, subtitles
	var banner_height = 100
	var radius = 35
	var font_size1 = 24
	var font_size2 = 18

	# Draws banner rectangle above the dection box by 50px
	var banner = Rect2(
		x,
		y - banner_height + 50,
		w,
		banner_height
	)

	# Makes the banner slightly transparent
	var panel_color = Color(
		hud_color.r,
		hud_color.g,
		hud_color.b,
		0.75
	)

	# Draws rounded rectangle background
	draw_panel(
		banner,
		panel_color,
		radius
	)

	# -------------------------
	# Common Name
	# -------------------------

	# Converts JSON data to Big Text
	var common = _resolve_field("Common Name", detection)

	# Measures text size to find pixel width of text
	var common_size = font.get_string_size(
		common,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		font_size1
	)

	# Centers text based on text width
	draw_string(
		font,
		Vector2(
			banner.position.x + banner.size.x/ - common_size.x+40,
			banner.position.y + 40
		),
		common,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		font_size1,
		Color.WHITE
	)

	# -------------------------
	# Scientific Name
	# -------------------------

	# Converts JSON data to Small Text
	var scientific = _resolve_field("Scientific Name", detection)

	# Measures text size to find pixel width of text
	var sci_size = font.get_string_size(
		scientific,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		font_size2
	)

	# Centers text based on text width
	draw_string(
		font,
		Vector2(
			banner.position.x + banner.size.x/ - sci_size.x+40,
			banner.position.y + 70
		),
		scientific,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		font_size2,
		Color.WHITE
	)

	# -------------------------
	# Confidence Circle

	# Positions circle: right side of banner and vertically centered
	var circle_center = Vector2(
		banner.position.x + banner.size.x - 50,
		banner.position.y + banner.size.y/2
	)

	# Create transparent circle and blends into the HUD color with a soft hightlight
	var circle_color = hud_color.lerp(Color.WHITE, 0.6)
	circle_color.a = 0.9

	# Draw circle
	draw_circle(
		circle_center,
		35,
		circle_color
	)

# Converts JSON data to Subtitle Text
	var confidence = str(
		round(float(detection.get("confidence",0))*100)
	) + "%"

	# Measures text size to find pixel width of text
	var conf_size = font.get_string_size(
		confidence,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		20
	)

	# Centers text based on text width
	draw_string(
		font,
		Vector2(
			circle_center.x - conf_size.x/2,
			circle_center.y + 8
		),
		confidence,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		20,
		Color.WHITE
	)

func draw_panel(rect: Rect2, color: Color, radius: int):
	
	# Draw style box
	var style := StyleBoxFlat.new()

	# Set background color
	style.bg_color = color

	# Set each corner rounding
	style.corner_radius_top_left = radius + 150
	style.corner_radius_top_right = radius + 150
	style.corner_radius_bottom_left = radius + 150
	style.corner_radius_bottom_right = radius + 150

	draw_style_box(style, rect)

# --------------------------------------------------------------
# LAYOUT 3 and Helper Functions
# --------------------------------------------------------------
func _draw_layout3(x, y, w, _h, detection, font):

	# Extract JSON data and convert into text
	var common = _resolve_field("Common Name", detection)
	var scientific = _resolve_field("Scientific Name", detection)

	var confidence = str(
		round(float(detection.get("confidence", 0.0)) * 100)
	) + "%"

	# -------------------------
	# Positioning
	# -------------------------
	# Define circle radius and line thickness
	var circle_radius = 40
	var line_thickness = 20

	# X is slightly shifted left/right, Y is top of detection box
	var circle_center = Vector2(
		x + circle_radius - 30,
		y 
	)

	# Stars a little below circle radius
	var line_start = Vector2(
		circle_center.x,
		circle_center.y + circle_radius - 25
	)

	# Sets line length 
	var line_end = Vector2(
		x + w,
		line_start.y
	)

	# -------------------------
	# Draw Circle
	# -------------------------
	draw_circle(
		circle_center,
		circle_radius,
		hud_color
	)

	# -------------------------
	# Draw Line
	# -------------------------
	# Connects circle to the line
	draw_line(
		line_start,
		line_end,
		hud_color,
		line_thickness
	)

	# -------------------------
	# Top Text
	# -------------------------
	# Finds how wide the text is for allignment
	var _common_size = font.get_string_size(
		common,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		28
	)

	# Finds how wide the text is for allignment
	var conf_size = font.get_string_size(
		confidence,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		28
	)

# Right alligns Common name (left) inside bouding box
	draw_string(
		font,
		Vector2(x + 60, y - 5),
		common,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		28,
		Color.BLACK
	)

	# Confidence (right) hugs right edge 
	draw_string(
		font,
		Vector2(
			x + w - conf_size.x,
			y - 5
		),
		confidence,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		28,
		Color.BLACK
	)

	# -------------------------
	# Scientific Name
	# -------------------------
	# Same allignment as common name; slightly lower (y + 22)and smaller fount (18)
	draw_string(
		font,
		Vector2(x + 60, y + 22),
		scientific,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		18,
		Color.BLACK
	)
