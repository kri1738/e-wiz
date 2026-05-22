from dotenv import load_dotenv
import os
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

class Message(BaseModel):
    message: str
    history: list

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/chat")
def chat(data: Message):
    # Keep only last 10 exchanges to avoid token limits
    history = [{"role": "system", "content": system_prompt}]
    history += data.history[-20:]
    history.append({"role": "user", "content": data.message})

    def stream():
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=history,
                stream=True
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"[ERROR] {str(e)}"

    return StreamingResponse(stream(), media_type="text/plain")

app.mount("/", StaticFiles(directory="static"), name="static")