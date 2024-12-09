import json
import asyncio
import websockets
import pyaudio


async def play_audio(queue: asyncio.Queue):
    """
    Asynchronously play audio chunks from the queue with initial buffering.

    Args:
        queue (asyncio.Queue): Queue containing audio chunks to be played.
    """
    player = pyaudio.PyAudio()
    player_stream = player.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    try:
        # Step 1: Initial buffering
        buffer = []
        while len(buffer) < 10:  # Collect initial 10 chunks for buffering
            audio_chunk = await queue.get()
            if audio_chunk is None:
                return  # Exit signal received
            buffer.append(audio_chunk)

        # Play buffered audio
        for chunk in buffer:
            player_stream.write(chunk)

        # Step 2: Play audio in real-time
        while True:
            audio_chunk = await queue.get()
            if audio_chunk is None:  # Exit signal
                return
            player_stream.write(audio_chunk)
    finally:
        player_stream.stop_stream()
        player_stream.close()
        player.terminate()


async def send_wav_file(url: str, file_path: str):
    """
    Send a .wav file to a WebSocket server and handle responses.

    Args:
        url (str): WebSocket server URL.
        file_path (str): Path to the .wav file to be sent.
    """
    audio_queue = asyncio.Queue()  # Queue for audio chunks

    # Start audio playback in a separate task
    audio_task = asyncio.create_task(play_audio(audio_queue))

    try:
        async with websockets.connect(url) as websocket:
            # Step 1: Read the .wav file in binary mode
            with open(file_path, "rb") as wav_file:
                audio_data = wav_file.read()

            # Step 2: Send the binary audio data to the server
            await websocket.send(audio_data)

            # Step 3: Listen for server responses
            while True:
                response = await websocket.recv()

                if isinstance(response, str):  # JSON response
                    data = json.loads(response)
                    if data["type"] == "STT":
                        print(f"STT Result: {data['content']}")
                    elif data["type"] == "LLM":
                        print(f"LLM Response: {data['content']}")
                    elif data["type"] == "END":
                        print("All processes completed. Closing WebSocket.")
                        break  # Exit the loop to close WebSocket
                elif isinstance(response, bytes):  # Audio response
                    await audio_queue.put(response)  # Add audio chunk to playback queue
    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed: {e}")
    finally:
        # Signal audio playback task to exit
        await audio_queue.put(None)
        await audio_task  # Wait for the audio playback task to finish


# Example usage
# Uncomment and replace `url` and `file_path` with actual values
# url = 'ws://example.com/roleplay/stream'
# file_path = "sample_input_0.wav"
# asyncio.run(send_wav_file(url, file_path))