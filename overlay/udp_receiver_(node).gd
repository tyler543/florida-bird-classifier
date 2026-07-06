extends Node

var udp := PacketPeerUDP.new()

@onready var overlay = $"../OverlayLayer (CanvasLayer)/DetectionOverlay (Control)"

func _ready():
	udp.bind(4242)
	print("Listening for AI detections")

func _process(_delta):
	while udp.get_available_packet_count() > 0:
		var packet = udp.get_packet()
		var text = packet.get_string_from_utf8()
		var data = JSON.parse_string(text)

		if typeof(data) != TYPE_DICTIONARY:
			continue

		print("RAW:", data)
		
		match data.get("type", ""):

			"detection":
				overlay.update_detections([data])

			"hud":
				overlay.update_hud_config(data)

			_:
				print("Unknown packet type:", data)
