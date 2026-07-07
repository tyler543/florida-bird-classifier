extends TextureRect

const FRAME_W = 960
const FRAME_H = 540
const FRAME_BYTES = FRAME_W * FRAME_H * 3
const PATH = "/tmp/latest_frame.raw"

var timer := 0.0
var cached_texture: ImageTexture = null

func _ready():
	size = get_viewport_rect().size
	position = Vector2.ZERO
	Input.set_mouse_mode(Input.MOUSE_MODE_HIDDEN)

func _process(delta):
	timer += delta
	if timer < 0.016:
		return
	timer = 0.0

	var file = FileAccess.open(PATH, FileAccess.READ)
	if file == null:
		return

	var bytes = file.get_buffer(FRAME_BYTES)
	file.close()

	if bytes.size() != FRAME_BYTES:
		return

	var image = Image.create_from_data(FRAME_W, FRAME_H, false, Image.FORMAT_RGB8, bytes)

	if cached_texture == null:
		cached_texture = ImageTexture.create_from_image(image)
		texture = cached_texture
	else:
		cached_texture.update(image)
