# J.A.R.V.I.S — Just A Rather Very Intelligent System

A fully offline, Iron Man-inspired AI assistant built with Python. Runs entirely on your local machine — no cloud, no API keys, no internet required.

---

## Features

- **Natural language commands** — talk to JARVIS like a person
- **App launcher** — opens apps and brings them to the foreground (`open chrome`, `launch spotify`)
- **Desktop automation** — types text, presses hotkeys, opens websites
- **JARVIS voice** — deep Microsoft David TTS with activation beeps and "Sir," personality
- **Voice input** — say "JARVIS" → it listens → Whisper transcribes → agent responds
- **System tools** — CPU/RAM/disk info, process list, kill processes
- **File tools** — read, write, list, search files
- **Git tools** — status, log, diff
- **Long-term memory** — SQLite-backed facts and conversation history
- **REST API** — FastAPI server at `http://127.0.0.1:8000`
- **Iron Man HUD** — animated arc reactor frontend at `http://127.0.0.1:8000/app`

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- A pulled Ollama model (default: `llama3.2:1b`)
- Windows 10/11 (voice features use Windows SAPI; API and tools work on all platforms)

---

## Quick Start

### 1. Install Ollama and pull a model

```bash
# Download from https://ollama.com and install, then:
ollama pull llama3.2:1b
```

> For better responses, use a larger model: `ollama pull mistral` or `ollama pull llama3.2:3b`

### 2. Clone and install dependencies

```bash
git clone <your-repo-url>
cd jarvis
pip install -r requirements.txt
```

### 3. Configure (optional)

Copy `.env.example` to `.env` and edit if needed:

```bash
cp .env.example .env
```

The main config file is `config/config.yaml` — you can change the model, voice settings, etc.

### 4. Run JARVIS

**API + Voice (full system):**
```bash
python main.py --skip-checks
```

**API only (no voice):**
```bash
python main.py --api-only --skip-checks
```

**Debug mode:**
```bash
python main.py --debug --skip-checks
```

Open the HUD in your browser: **http://127.0.0.1:8000/app**

---

## Usage

### Via the HUD (browser)

Open **http://127.0.0.1:8000/app** — you get the Iron Man-style interface with:

- Type commands in the input box and press **Enter** or click **EXECUTE**
- Click the **🎤** button or press **V** for browser voice input
- The arc reactor animates: blue=standby, green=listening, gold=processing
- Right panel shows system status, memory facts, and performance metrics

### Via the API

```bash
# Send a command
curl -X POST http://127.0.0.1:8000/command \
  -H "Content-Type: application/json" \
  -d '{"input": "open chrome"}'

# Health check
curl http://127.0.0.1:8000/health

# API docs
open http://127.0.0.1:8000/docs
```

### Command Examples

| What you say | What happens |
|---|---|
| `open chrome` | Chrome opens and comes to foreground |
| `launch spotify` | Spotify opens maximized |
| `open notepad` | Notepad opens in foreground |
| `open calculator` | Calculator pops up |
| `type hello world` | Types text at current cursor position |
| `press hotkey ctrl+c` | Copies selection |
| `press hotkey win+d` | Shows desktop |
| `open website youtube.com` | Opens in default browser |
| `system status` | Returns CPU, RAM, disk usage |
| `list processes` | Shows top running processes |
| `what files are in this folder` | Lists current directory |
| `git status` | Shows git working tree status |

### Voice Commands (say "JARVIS" to activate)

1. Start JARVIS with `python main.py --skip-checks`
2. Say **"JARVIS"** — you'll hear a beep and "Yes, Sir?"
3. Speak your command
4. JARVIS responds with David's voice

---

## Project Structure

```
jarvis/
├── main.py                  # Entry point
├── config/
│   └── config.yaml          # All settings (model, voice, tools, etc.)
├── agents/
│   ├── engine.py            # AgentEngine — main orchestrator
│   ├── executor.py          # ReAct loop with keyword fast-path
│   ├── planner.py           # Multi-step planner (multi-agent mode)
│   ├── supervisor.py        # Response quality checker
│   └── base.py              # BaseAgent + AgentResult
├── tools/
│   ├── system_tools.py      # open_app, get_system_info, list_processes
│   ├── desktop_tools.py     # type_text, press_hotkey, open_website
│   ├── file_tools.py        # read_file, write_file, list_dir
│   ├── git_tools.py         # git_status, git_log, git_diff
│   ├── email_tool.py        # send_email (configure SMTP in .env)
│   └── registry.py          # Tool registry and execution engine
├── voice/
│   ├── tts.py               # JARVIS TTS (pyttsx3 + David voice + beeps)
│   ├── stt.py               # Whisper speech-to-text
│   ├── wake_word.py         # "JARVIS" wake word detection
│   └── pipeline.py          # Full voice loop
├── memory/
│   ├── short_term.py        # Rolling message window (in-memory)
│   ├── long_term.py         # SQLite facts + conversation history
│   └── manager.py           # MemoryManager facade
├── api/
│   ├── server.py            # FastAPI app factory
│   ├── routes/              # /command, /health, /memory endpoints
│   └── middleware.py        # Request logging + metrics
├── core/
│   └── llm.py               # Ollama LLM client
├── utils/
│   ├── logger.py            # Structured JSON logging
│   ├── metrics.py           # In-process counters + latency histograms
│   ├── startup.py           # 5 startup health checks
│   └── shutdown.py          # LIFO graceful shutdown hooks
├── frontend/
│   └── index.html           # Iron Man HUD (served at /app)
└── tests/                   # pytest test suite
```

---

## Configuration

Edit `config/config.yaml`:

```yaml
llm:
  model: "llama3.2:1b"    # change to mistral, llama3.2:3b, etc.
  temperature: 0.7

voice:
  enabled: true
  tts:
    rate: 155             # speech rate (words per minute)
  wake_word:
    keyword: "jarvis"     # word to listen for

tools:
  system:
    enabled: true
    allow_shell_exec: false   # set true to enable kill_process
```

To use a better model:
```bash
ollama pull mistral
# Then change config.yaml: model: "mistral"
```

---

## App Aliases

JARVIS understands natural names for apps. Examples:

| Say | Opens |
|---|---|
| `chrome` / `google chrome` | Google Chrome |
| `edge` / `microsoft edge` | Microsoft Edge |
| `vscode` / `vs code` | Visual Studio Code |
| `terminal` / `windows terminal` | Windows Terminal |
| `word` / `excel` / `powerpoint` | Microsoft Office apps |
| `discord` / `slack` / `teams` | Communication apps |
| `spotify` / `vlc` / `steam` | Entertainment apps |

---

## Running Tests

```bash
# Run all tests
pytest

# Run specific module
pytest tests/test_api.py -v
pytest tests/test_tools.py -v

# Skip slow tests
pytest -m "not slow"
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info and endpoint list |
| `GET` | `/health` | LLM health check |
| `GET` | `/status` | Uptime, model, memory stats |
| `GET` | `/metrics` | Performance counters and latency |
| `POST` | `/command` | Execute a natural-language command |
| `GET` | `/memory/facts` | List all stored facts |
| `POST` | `/memory/facts` | Store a fact `{"key": "...", "value": "..."}` |
| `DELETE` | `/memory/facts/{key}` | Delete a fact |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/app` | Iron Man HUD frontend |

---

## Troubleshooting

**Server not responding**
```bash
# Make sure Ollama is running
ollama serve
# Then restart JARVIS
python main.py --api-only --skip-checks
```

**Apps not opening**
- Make sure the app is installed
- Try the full app name: `open google chrome` instead of `open chrome`
- Check `config/config.yaml` → `tools.system.enabled: true`

**Voice not working**
- Run `python main.py --skip-checks` (not `--api-only`)
- Check your microphone is connected and set as default
- Whisper model downloads on first use (~150 MB for `base`)

**Slow responses**
- Switch to a smaller model: `model: "llama3.2:1b"` in config.yaml
- Or use a faster machine / GPU

**"Model not found" error**
```bash
ollama pull llama3.2:1b
```

---

## Built With

- [Ollama](https://ollama.com) — local LLM inference
- [FastAPI](https://fastapi.tiangolo.com) — REST API
- [Whisper](https://github.com/openai/whisper) — offline speech recognition
- [pyttsx3](https://pypi.org/project/pyttsx3/) — offline text-to-speech
- [pyautogui](https://pyautogui.readthedocs.io) — desktop automation
- [psutil](https://psutil.readthedocs.io) — system monitoring

---

## License

MIT License — use it, modify it, make it your own.
