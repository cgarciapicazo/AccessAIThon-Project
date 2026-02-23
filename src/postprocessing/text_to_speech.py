import asyncio
import io
import random
import threading
import queue


import edge_tts
from edge_tts import VoicesManager
import soundfile as sf
import sounddevice as sd

class TTSService:
   
    def __init__(self, gender="Male", language="en", rate="+0%"):
        self.gender = gender
        self.language = language
        self.rate = rate

        self._q = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, daemon=False)
        self._thread.start()

    def speak(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        self._q.put(text)

    def close(self):
        self._stop.set()
        self._q.put(None)
        try:
            self._thread.join(timeout=2)
        except Exception:
            pass


    def _thread_main(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._worker(loop))
        try:
            loop.close()
        except Exception:
            pass


    async def _worker(self, loop):
        voices = await VoicesManager.create()
        voice_choices = voices.find(Gender=self.gender, Language=self.language)
        if not voice_choices:
            voice_choices = voices.find(Language=self.language)


        while not self._stop.is_set():
            text = await loop.run_in_executor(None, self._q.get)
            if text is None:
                break


            try:
                await self._speak_once(text, voice_choices)
            except Exception as e:
                print(f"TTS error: {e}")


            try:
                self._q.task_done()
            except Exception:
                pass

    async def _speak_once(self, text, voice_choices):
        voice_name = random.choice(voice_choices)["Name"]
        communicate = edge_tts.Communicate(text, voice_name, rate=self.rate)


        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])


        audio_buffer.seek(0)
        data, samplerate = sf.read(audio_buffer, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()