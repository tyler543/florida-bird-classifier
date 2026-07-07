extends TextureRect

var timer := 0.0
var path = "/tmp/latest_frame.jpg"
var cached_texture: ImageTexture = null

func _process(delta):
	timer += delta
	if timer < 0.1:
		return
	timer = 0.0

	var file = FileAccess.open(path, FileAccess.READ)
	if file == null:
		return

	var bytes = file.get_buffer(file.get_length())
	file.close()

	var image = Image.new()
	if image.load_jpg_from_buffer(bytes) != OK:
		return

	if cached_texture == null:
		cached_texture = ImageTexture.create_from_image(image)
		texture = cached_texture
	else:
		cached_texture.update(image)
