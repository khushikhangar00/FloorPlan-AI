# Floorplan AI (Streamlit)

Generate customizable office/interior floor plans from a text brief. The model
returns a structured layout (rooms, doors, furniture) which is rendered as an
SVG blueprint you can then edit room-by-room and export.

## Project structure

```
floorplan-streamlit/
  app.py              Streamlit UI
  floorplan_core.py   Prompting, model calls, JSON parsing, SVG rendering (no Streamlit deps)
  requirements.txt
  .streamlit/secrets.toml.example
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Using Claude

Paste an Anthropic API key into the sidebar field, or avoid re-entering it by
copying `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and
filling in `ANTHROPIC_API_KEY`. That file is git-ignored by default on
Streamlit Community Cloud when added via the app's Secrets settings — don't
commit real keys.

## Using an open-source model instead

Switch "Model backend" to "Custom endpoint" in the sidebar and point it at
any OpenAI-compatible `/v1/chat/completions` server:

- **Ollama**: `ollama serve`, then `ollama pull llama3.1:8b`, endpoint
  `http://localhost:11434/v1/chat/completions`, model `llama3.1:8b`.
- **vLLM**: `vllm serve <model>` exposes an OpenAI-compatible server by
  default, usually `http://localhost:8000/v1/chat/completions`.
- **LM Studio / text-generation-inference**: same idea — check the tool's
  docs for its OpenAI-compatible port.

No code changes needed; `floorplan_core.call_openai_compatible()` handles it.
Smaller open-source models are less reliable at returning strict JSON — if
generation fails, try a larger model or lower the room count in the prompt.

## Deploying

**Streamlit Community Cloud** (free, easiest):
1. Push this folder to a GitHub repo.
2. On share.streamlit.io, "New app" → point at the repo, `app.py` as the entry file.
3. In the app's Settings → Secrets, add `ANTHROPIC_API_KEY = "sk-ant-..."` if using Claude.

**Docker**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

If self-hosting alongside Ollama, run both containers on the same Docker
network and set the endpoint URL to the Ollama container's hostname instead
of `localhost`.

## Known limitations

- Layout is generated in one LLM call with no geometric constraint solving,
  so occasional room overlap is possible on complex briefs (the prompt asks
  the model to avoid it, but it isn't guaranteed).
- Furniture is edited via numeric sliders rather than drag-and-drop — plain
  Streamlit has no native canvas drag support. `streamlit-elements` or a
  custom component could add that later.
- Smaller/local models may return malformed JSON more often than Claude;
  the app surfaces the parse error rather than guessing.
