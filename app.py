from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from routers.chat import router as chat_router

app = FastAPI(title="My Chatbot API")
app.include_router(chat_router)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

