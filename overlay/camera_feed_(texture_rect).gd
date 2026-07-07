extends TextureRect

var timer := 0.0
var path = "/tmp/latest_frame.jpg"

func _process(delta):
	timer += delta

	if timer < 0.1:
		return

	timer = 0.0

	if not FileAccess.file_exists(path):
		return

	var image = Image.load_from_file(path)
	if image != null:
		texture = ImageTexture.create_from_image(image)
