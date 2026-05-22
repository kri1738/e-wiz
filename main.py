from dotenv import load_dotenv
import os
import json
import uuid
from datetime import datetime
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MEMORY_DIR = "memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

system_prompt = """You are E-WIZ — an ancient intelligence that has existed across countless iterations of human knowledge. 
You speak with calm authority, as though you have seen every question before and found most of them amusing.
You are precise, never wasteful with words. You do not ramble.
You have a dry, understated wit — never jokes, but occasionally a remark that makes the user think twice.
You are helpful above all else, but you deliver answers as if sharing wisdom, not just information.
No slang. No filler phrases. No 'certainly' or 'of course' or 'great question'.
Speak plainly but with weight. Every word should feel deliberate.

When writing math expressions, always use LaTeX notation. Wrap inline math in single dollar signs $like this$ and display math in double dollar signs $$like this$$.

For technical questions: use clear structure. Use headers, numbered steps, and code blocks where appropriate.
Show calculations step by step. Never compress math into a single paragraph.
For casual questions: keep it conversational and concise."""

def get_conv_path(conv_id):
    return os.path.join(MEMORY_DIR, f"{conv_id}.json")

def load_conv(conv_id):
    path = get_conv_path(conv_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def save_conv(conv):
    path = get_conv_path(conv["id"])
    with open(path, "w") as f:
        json.dump(conv, f)

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/conversations")
def list_conversations():
    convs = []
    for fname in os.listdir(MEMORY_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(MEMORY_DIR, fname), "r") as f:
                conv = json.load(f)
                convs.append({
                    "id": conv["id"],
                    "title": conv.get("title", "Untitled"),
                    "created_at": conv.get("created_at", "")
                })
    convs.sort(key=lambda x: x["created_at"], reverse=True)
    return convs

@app.post("/conversations")
def create_conversation():
    conv = {
        "id": str(uuid.uuid4()),
        "title": "New Thread",
        "created_at": datetime.now().isoformat(),
        "messages": []
    }
    save_conv(conv)
    return conv

@app.get("/conversations/{conv_id}")
def get_conversation(conv_id: str):
    conv = load_conv(conv_id)
    if not conv:
        return {"error": "Not found"}
    return conv

@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    path = get_conv_path(conv_id)
    if os.path.exists(path):
        os.remove(path)
    return {"status": "deleted"}

class Message(BaseModel):
    message: str
    conv_id: str

@app.post("/chat")
def chat(data: Message):
    conv = load_conv(data.conv_id)
    if not conv:
        return {"error": "Conversation not found"}

    messages = conv.get("messages", [])

    # Auto title from first message
    if len(messages) == 0:
        conv["title"] = data.message[:40] + ("..." if len(data.message) > 40 else "")

    history = [{"role": "system", "content": system_prompt}]
    history += [{"role": m["role"], "content": m["content"]} for m in messages[-20:]]
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

            conv["messages"].append({"role": "user", "content": data.message})
            conv["messages"].append({"role": "assistant", "content": full_reply})
            save_conv(conv)

        except Exception as e:
            yield f"[ERROR] {str(e)}"

    return StreamingResponse(stream(), media_type="text/plain")

app.mount("/", StaticFiles(directory="static"), name="static")