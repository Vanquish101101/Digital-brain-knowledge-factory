import pyttsx3

from kf.transcribe import transcribe_audio


def test_transcribes_synthesized_speech(tmp_path):
    audio_path = tmp_path / "speech.wav"
    engine = pyttsx3.init()
    engine.save_to_file("testing one two three", str(audio_path))
    engine.runAndWait()

    text = transcribe_audio(
        audio_path, model_size="small", cache_dir=str(tmp_path / "model-cache"), language="en"
    )

    assert "test" in text.lower()
