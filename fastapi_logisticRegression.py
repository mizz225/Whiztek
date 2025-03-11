from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import re
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np

loaded_pca = joblib.load("./saved_files/pca_9123_pp.pkl")

sbert_model = SentenceTransformer('./saved_files/MiniLM_l6_v2')

loaded_logreg = joblib.load("./saved_files/logreg_model_9123_pp.pkl")

def clean_text(text):
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_text_sentiment(text):
    text = clean_text(text)
    inp = np.array([sbert_model.encode(text)])
    inp = loaded_pca.transform(inp)
    y_pred = loaded_logreg.predict(inp)
    if y_pred[0]==1:
        return "Positive"
    else:
        return "Negative"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def home():
    with open('templates/index4.html', 'r') as file:
        return HTMLResponse(content=file.read())


@app.post("/predict")
# async def get_top_news(keyword: str = Form(...)):
async def get_top_news(request: Request):
    try:
        form_data = await request.form()
        comment = form_data.get('comment')
        pred_sent = predict_text_sentiment(comment)
        jsonResp = [{'sentiment':pred_sent}]
        return JSONResponse(content=jsonResp)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error occurred!")

# Command to run: uvicorn app_2:app --reload

if __name__ == "__main__":
    uvicorn.run("fastapi_logisticRegression:app", host="127.0.0.1", port=8002, reload=True)