import os
import asyncio
import logging
import sqlite3

import asqlite
from obswebsocket import obsws, requests
import twitchio
from twitchio import eventsub
from twitchio.ext import commands

from dotenv import load_dotenv

load_dotenv()

# ── OBS Configuration ────────────────────────────────────────────────────────────

OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = int(os.getenv("OBS_PORT", 4455))
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "")
OBS_IMAGE_SOURCE = "MemeImage"  # Name of your Image Source in OBS
OBS_AUDIO_SOURCE = "MemeAudio"  # Name of your Media (audio) Source in OBS
REACT_DELAY = 30  # Seconds to keep meme on-screen


# Create a single, global OBS client
obs_client = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)


def connect_obs():
    try:
        obs_client.connect()
        print("✅ Connected to OBS.")
    except Exception as e:
        print(f"❌ OBS connect error: {e}")


def disconnect_obs():
    try:
        obs_client.disconnect()
        print("👋 Disconnected from OBS.")
    except Exception as e:
        print(f"❌ OBS disconnect error: {e}")


# ── Meme Definitions ────────────────────────────────────────────────────────────

ASSET_FOLDER = os.path.join(os.path.dirname(__file__), "assets")

MEMES_KEYS = {
    "cinema": ["cinema"],
    "uiia": ["dance", "dinodance", "uiia"],
}


def get_scene_item_id(scene_name, source_name):
    items = obs_client.call(
        requests.GetGroupSceneItemList(sceneName=scene_name)
    ).getSceneItems()
    print(f"Group items: {items}")
    for item in items:
        if item["sourceName"] == source_name:
            return item["sceneItemId"]
    return None


SOURCE_IDS = {}
MEMES = {
    "cinema": {
        "image": "normal.webp",
        "audio": "normal.mp3",
        "active": False,
        "live": 1,
    },
    "uiia": {
        "image": "uiia.gif",
        "audio": "uiia.mp3",
        "active": False,
        "live": 1,
    },
}

# ── OBS Helper ──────────────────────────────────────────────────────────────────


async def trigger_meme(name: str):
    """Show image + play audio for MEMES[name], then hide/stop after its own 'live' duration."""
    print(f"Triggering meme: {name}")
    meme = MEMES[name]
    if meme["active"]:
        return  # already running

    meme["active"] = True
    image_path = os.path.join(ASSET_FOLDER, name, meme["image"])
    audio_path = os.path.join(ASSET_FOLDER, name, meme["audio"])
    duration = meme.get("live", REACT_DELAY)  # fallback if live is missing

    loop = asyncio.get_running_loop()

    # 1) Point the OBS Image source at the new file
    await loop.run_in_executor(
        None,
        lambda: obs_client.call(
            requests.SetInputSettings(
                inputName=OBS_IMAGE_SOURCE,
                inputSettings={"file": image_path},
                overlay=True,
            )
        ),
    )

    # 2) Show the image
    await loop.run_in_executor(
        None,
        lambda: obs_client.call(
            requests.SetSceneItemEnabled(
                sceneItemId=SOURCE_IDS["MemeImage"],
                sceneItemEnabled=True,
            )
        ),
    )

    # 3) Point the Audio source and play
    await loop.run_in_executor(
        None,
        lambda: obs_client.call(
            requests.SetInputSettings(
                inputName=OBS_AUDIO_SOURCE,
                inputSettings={"local_file": audio_path},
                overlay=True,
            )
        ),
    )
    await loop.run_in_executor(
        None,
        lambda: obs_client.call(
            requests.PlayPauseMedia(
                inputName=OBS_AUDIO_SOURCE,
                playPause=False,  # false = play
                restart=True,
            )
        ),
    )

    # 4) Wait exactly `live` seconds
    await asyncio.sleep(duration)

    # 5) Hide image & stop audio
    await loop.run_in_executor(
        None,
        lambda: obs_client.call(
            requests.SetSceneItemEnabled(
                sceneItemId=SOURCE_IDS["MemeImage"],
                sceneItemEnabled=False,
            )
        ),
    )
    await loop.run_in_executor(
        None,
        lambda: obs_client.call(
            requests.PlayPauseMedia(
                inputName=OBS_AUDIO_SOURCE, playPause=True  # true = pause/stop
            )
        ),
    )

    meme["active"] = False


LOGGER = logging.getLogger("Bot")


async def main_async():
    connect_obs()

    try:
        global SOURCE_IDS
        SOURCE_IDS = {
            "MemeImage": get_scene_item_id("Effects", "MemeImage"),
            "MemeAudio": get_scene_item_id("Effects", "MemeAudio"),
        }
        print(f"Source IDs: {SOURCE_IDS}")
        await trigger_meme("uiia")
    except KeyboardInterrupt:
        disconnect_obs()
        LOGGER.warning("Shutdown via KeyboardInterrupt")


if __name__ == "__main__":
    # this will start an event loop, run main_async(), then tear it down
    asyncio.run(main_async())
