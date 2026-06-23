from fastapi import FastAPI
from backend.routes.analyze import router

app = FastAPI(title="Behavior AI System")

app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Behavior AI Backend Running"}