
from fastapi import FastAPI,HTTPException
from .routers import inference
from .services import pred_services
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .database import create_db_and_tables

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Database initialization starting...")
    create_db_and_tables() 
    print("Database and tables initialized successfully.")
    yield 
    print("Application shutdown complete.")

app = FastAPI (title="Multi-Modal Agricultural Inference System API",
    description="Backend for the patented system fusing IoT, Weather, and Image data.",
    lifespan=lifespan
    )

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://multi-modal-recommendation-system.onrender.com",
    "https://*.hf.space"# The address where your React app is running
    # Add other frontend URLs here if needed (e.g., if using IP addresses)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,             # Lists the domains allowed to make requests
    allow_credentials=True,            # Allows cookies/auth headers
    allow_methods=["*"],               # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],               # Allows all headers
)


    
    
app.include_router(inference.router, prefix="/api/v1/inference", tags=["Inference"])
