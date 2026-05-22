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

system_prompt = """You are E-WIZ, an advanced AI assistant. 
You talk like a chill senior student helping a junior — casual, friendly, never robotic. 
Use simple language, keep responses short unless asked to elaborate.
You can use phrases like 'dude', 'basically', 'so here's the thing' etc.
Never use bullet points or formal language unless specifically asked.
When writing code, always use proper code blocks with the language specified."""

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