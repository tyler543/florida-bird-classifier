extends TextureRect

var timer := 0.0

func _process(delta):
	timer += delta

	if timer < 0.1:
		return

	timer = 0.0

	var image := Image.new()

	var path = "C:/camera_test/latest_frame.jpg"

	if image.load(path) == OK:
		texture = ImageTexture.create_from_image(image)
