import json
import asyncio
import websockets
import pyaudio

async def play_audio(queue):
    """Asynchronously play audio chunks from the queue with initial buffering."""
    player = pyaudio.PyAudio()
    player_stream = player.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    # Step 1: Initial buffering
    buffer = []
    while len(buffer) < 10:  # Collect initial 10 chunks for buffering
        audio_chunk = await queue.get()
        if audio_chunk is None:
            break
        buffer.append(audio_chunk)

    # Play buffered audio
    for chunk in buffer:
        player_stream.write(chunk)

    # Step 2: Play audio in real-time
    while True:
        audio_chunk = await queue.get()
        if audio_chunk is None:  # Exit signal
            break
        player_stream.write(audio_chunk)
    
    player_stream.stop_stream()
    player_stream.close()
    player.terminate()


async def send_wav_file(url, file_path):
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
            # print("Audio data sent to server.")

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
    finally:
        # Signal audio playback task to exit
        await audio_queue.put(None)
        await audio_task  # Wait for the audio playback task to finish

# Run the client
# url = ''
# file_path = "sample_input_0.wav"
# asyncio.run(send_wav_file(url, file_path))

# file_path = "sample_input_1.wav"
# asyncio.run(send_wav_file(url, file_path))

# import requests
# print(requests.post(url))