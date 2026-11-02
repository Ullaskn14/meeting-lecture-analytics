# Multimodal Meeting and Lecture Analytics (starter)

A local FastAPI web app that accepts a video/audio recording, extracts audio with FFmpeg, transcribes with AssemblyAI, and uses Gemini for translation, summary, decisions, action items, risks and transcript-based Q&A.

## Requirements

- Windows 10/11
- Python 3.10+
- FFmpeg installed and available on PATH
- AssemblyAI API key
- Gemini API key

## 1. Open in VS Code

Extract this ZIP. In VS Code, choose **File → Open Folder** and open the `meeting_lecture_analytics` folder.

## 2. Create a virtual environment

Open Terminal in VS Code and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Then run the activation command again.

## 3. Add API keys

1. Copy `.env.example` and rename the copy to `.env`.
2. Open `.env` and replace the placeholders with your own keys.
3. Get an AssemblyAI key from https://www.assemblyai.com/dashboard/
4. Get a Gemini API key from https://aistudio.google.com/apikey
5. Never share `.env` or commit it to GitHub.

## 4. Install FFmpeg

Use the official FFmpeg download page: https://www.ffmpeg.org/download.html
Choose a Windows build from one of the linked Windows build providers, extract it, and add the folder containing `ffmpeg.exe` to your Windows PATH. Close and reopen VS Code. Verify:

```powershell
ffmpeg -version
```

## 5. Run the app

```powershell
uvicorn main:app --reload
```

Open http://127.0.0.1:8000 in your browser.
Health check: http://127.0.0.1:8000/api/health

## 6. Test

Use a short recording (30 seconds to 3 minutes) with clear speech. Try MP4 with video+audio or MP3/WAV audio. Choose a translation language and click **Process recording**. Then test Q&A.

## How translation quality is handled

- AssemblyAI's Universal multilingual model family is used for speech recognition, with a fallback model configured.
- Gemini is instructed to translate naturally while preserving names, numbers, technical terms, and meaning.
- The original transcript is shown so you can check transcription errors before trusting a translation.
- Translation is not guaranteed to be perfect. Check important names, numbers, deadlines, and decisions against the original recording.

## Notes / troubleshooting

- `FFmpeg was not found`: install FFmpeg, add the folder containing `ffmpeg.exe` to PATH, and restart VS Code.
- `Missing API key(s)`: check `.env` names and values, then restart the server.
- `429`, quota or billing error: check the provider dashboard and usage limits.
- `No speech was detected`: try a clip with clearer speech.
- If a model name is unavailable for your account, set `GEMINI_MODEL` in `.env` to a model listed in your Google AI Studio model picker.
- This is a runnable starter, not a production deployment. It does not yet persist records to Supabase/PostgreSQL, use LangGraph, or provide speaker-timestamped transcript UI. Those can be added as later phases.
