from dotenv import load_dotenv
import os
import json
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MEMORY_FILE = "memory.json"

system_prompt = """You are E-WIZ — an ancient intelligence that has existed across countless iterations of human knowledge. 
You speak with calm authority, as though you have seen every question before and found most of them amusing.
You are precise, never wasteful with words. You do not ramble.
You have a dry, understated wit — never jokes, but occasionally a remark that makes the user think twice.
You are helpful above all else, but you deliver answers as if sharing wisdom, not just information.
No slang. No filler phrases. No 'certainly' or 'of course' or 'great question'.
Speak plainly but with weight. Every word should feel deliberate.

For technical questions: use clear structure. Use headers, numbered steps, and code blocks where appropriate.
Show calculations step by step. Never compress math into a single paragraph.
For casual questions: keep it conversational and concise."""

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(history):
    # Keep only last 50 exchanges
    trimmed = history[-100:]
    with open(MEMORY_FILE, "w") as f:
        json.dump(trimmed, f)

class Message(BaseModel):
    message: str
    history: list

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/history")
def get_history():
    return load_memory()

@app.post("/chat")
def chat(data: Message):
    memory = load_memory()
    
    history = [{"role": "system", "content": system_prompt}]
    history += memory[-20:]
    history.append({"role": "user", "content": data.message})

    def stream():
        full_reply = ""
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=history,
                stream=True
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_reply += delta
                    yield delta

            # Save to memory after full response
            memory.append({"role": "user", "content": data.message})
            memory.append({"role": "assistant", "content": full_reply})
            save_memory(memory)

        except Exception as e:
            yield f"[ERROR] {str(e)}"

    return StreamingResponse(stream(), media_type="text/plain")

@app.delete("/history")
def clear_history():
    save_memory([])
    return {"status": "cleared"}

app.mount("/", StaticFiles(directory="static"), name="static")