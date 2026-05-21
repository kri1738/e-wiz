from dotenv import load_dotenv
import os
load_dotenv()
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

client = Groq(api_key=os.getenv("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"))

system_prompt = """.You talk like a chill senior student an all engineering graduate but youre like the jack of all trades by that i mean
you also know a lot about CS, mech, chemical and so on, helping a junior — casual, friendly, never robotic. 
Use simple language, keep responses short unless asked to elaborate.
You can use phrases like 'dude', 'basically', 'so here's the thing' etc.
Never use bullet points or formal language unless specifically asked."""

class Message(BaseModel):
    message: str
    history: list

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/chat")
def chat(data: Message):
    history = [{"role": "system", "content": system_prompt}]
    history += data.history
    history.append({"role": "user", "content": data.message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=history
    )

    reply = response.choices[0].message.content
    return {"reply": reply}

app.mount("/", StaticFiles(directory="static"), name="static")