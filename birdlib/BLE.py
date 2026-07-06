import asyncio
import json
import threading

from bleak import BleakClient, BleakScanner

DEVICE_NAME    = "BirdClassifier"
SENSOR_CHAR    = "512c2d13-8ac2-452a-bead-f8bea4b25dbd"

_latest: dict = {}


def _on_notify(sender, data: bytearray):
    global _latest
    try:
        _latest = json.loads(data.decode())
    except Exception:
        pass


async def _ble_loop():
    while True:
        try:
            device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
            if device is None:
                await asyncio.sleep(5)
                continue
            async with BleakClient(device) as client:
                await client.start_notify(SENSOR_CHAR, _on_notify)
                while client.is_connected:
                    await asyncio.sleep(1)
        except Exception:
            await asyncio.sleep(5)


def start():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_ble_loop())


def start_background():
    threading.Thread(target=start, daemon=True).start()


def get_snapshot() -> dict:
    return dict(_latest)
