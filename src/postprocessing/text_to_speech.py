import asyncio
import random
import io

import edge_tts
from edge_tts import VoicesManager
import soundfile as sf
import sounddevice as sd

TEXT = "You either die a hero, or you live long enough to see yourself become the villain. Why do we fall, Bruce? So we can learn to pick ourselves up. You think darkness is your ally. But you merely adopted the dark; I was born in it, molded by it. "


async def amain() -> None:
    """Main function"""
    voices = await VoicesManager.create()
    voice = voices.find(Gender="Male", Language="en")
    # Also supports Locales
    # voice = voices.find(Gender="Female", Locale="es-AR")

    communicate = edge_tts.Communicate(TEXT, random.choice(voice)["Name"])
    
    audio_buffer = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    audio_buffer.seek(0)

    data, samplerate = sf.read(audio_buffer)
    sd.play(data, samplerate)
    sd.wait()

if __name__ == "__main__":
    asyncio.run(amain())




"""
real time streaming playback example.

 async for chunk in communicate.stream():
    if chunk["type"] == "audio":
        play_chunk(chunk["data"])

"""