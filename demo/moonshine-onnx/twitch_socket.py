import os
import asyncio
from twitchio.ext import commands
from obswebsocket import obsws, requests
from dotenv import load_dotenv

load_dotenv()


# OBS-websocket client configuration.
OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = os.getenv("OBS_PORT", 4455)  # Default port for OBS-websocket is 4455.
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "")  # Password for OBS-websocket, if set.
OBS_SOURCE = "LiveReact"
# Create a global OBS websocket client.
obs_client = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)


def connect_obs():
    try:
        obs_client.connect()
        print("Connected to OBS via Websocket.")
    except Exception as e:
        print(f"Error connecting to OBS: {e}")


def disconnect_obs():
    try:
        obs_client.disconnect()
        print("Disconnected from OBS.")
    except Exception as e:
        print(f"Error disconnecting from OBS: {e}")


def main():
    try:
        # Connect to OBS
        connect_obs()

    except KeyboardInterrupt:
        print("Exiting...")
        disconnect_obs()


if __name__ == "__main__":
    main()
