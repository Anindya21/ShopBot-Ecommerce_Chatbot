from fastapi import FastAPI
from fastapi.responses import JSONResponse
from core.config import settings
from app.graph import ChatState
from pathlib import Path
from pydantic import BaseModel
import asyncio
import sys


current_dir = Path(__file__).parent
sys.path.append(str(Path(__file__).resolve().parent))

app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

chat_graph = ChatState()

class ChatRequest(BaseModel):
    conversation_id:str
    user_message:str
    history: list= []


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):

    try:
        input_state={
            "conversation_id": request.conversation_id,
            "messages": request.history + [{"role": "user", "content": request.user_message}],
            "context": None,
            "query": None,
            "needs_rag": False,
        }

        final_state= await asyncio.to_thread(chat_graph.invoke, input_state)
        bot_reply= final_state["messages"][-1]["content"]

        return {"reply": bot_reply}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "Welcome to the Chatbot API"}