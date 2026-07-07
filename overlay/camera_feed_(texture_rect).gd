extends TextureRect

var timer := 0.0

func _process(delta):
	timer += delta

	if timer < 0.1:
		return

	timer = 0.0

	var image := Image.new()

	var path = "/tmp/latest_frame.jpg"

	if FileAccess.file_exists(path) and image.load(path) == OK:
		texture = ImageTexture.create_from_image(image)
