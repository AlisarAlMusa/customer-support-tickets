from fastapi import FastAPI

from app.logging_setup import setup_logging
from app.routers import answer, ingest, predict, validate
from fastapi.middleware.cors import CORSMiddleware


setup_logging()


app = FastAPI(title="Customer Support Tickets API")

# ✅ ADD THIS HERE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(predict.router)
app.include_router(answer.router)
app.include_router(validate.router)


@app.get("/")
def read_root():
    return {"message": "Customer Support Tickets API"}
