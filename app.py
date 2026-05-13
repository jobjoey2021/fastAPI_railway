from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from services.stock import stock_gpt
from services.ai import stock_gpt

app = FastAPI()

# 允許 Quasar 前端呼叫
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Stock GPT API"}

@app.get("/analyze/{stock_id}")
def analyze_stock(stock_id: str):
    try:
        result = stock_gpt(stock_id)
        return {
            "success": True,
            "stock_id": stock_id,
            "report": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
