from fastapi import FastAPI

app = FastAPI(title="Personal Chatbot")


@app.get("/")
def root():
    return {"message": "Welcome to the Chatbot API"}