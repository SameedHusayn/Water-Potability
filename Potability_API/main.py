from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

# Create an instance of the FastAPI class
app = FastAPI()
app.mount("/static", StaticFiles(directory="D:\\Code\\Water Potability\\Water-Potability\\Potability_API\\static"), name="static")

# Load the SVM model from the pickle fileho
model = joblib.load("D:\\Code\\Water Potability\\Water-Potability\\Potability_API\\model.pkl")

# Create a Pydantic model to define the request payload
class WaterPotabilityRequest(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# Create an instance of the Jinja2Templates for rendering HTML templates
templates = Jinja2Templates(directory=r"D:\Code\Water Potability\Water-Potability\Potability_API\templates")

# Define a function to make predictions
def predict_potability(features: WaterPotabilityRequest):
    # Convert input features to a NumPy array
    input_data = np.array(list(features.dict().values())).reshape(1, -1)
    
    # Create a new StandardScaler instance
    scaler = StandardScaler()
    
    # Fit the scaler to the input data (assuming you have enough data to fit it)
    scaler.fit(input_data)
    
    # Apply scaling to the input data
    scaled_input_data = scaler.transform(input_data)
    
    # Make predictions using the loaded SVM model
    prediction = model.predict(scaled_input_data)
    
    # Return the predicted potability as 0 or 1 (0 for not potable, 1 for potable)
    return int(prediction[0])

# Create an endpoint to serve the HTML input form
@app.get("/")
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

# Create an endpoint to receive POST requests with input data
@app.post("/predict")
async def predict(request: Request, json_data: str = Form(...)):
    features = json.loads(json_data)
    # Call the predict_potability function to make predictions
    prediction = predict_potability(WaterPotabilityRequest(**features))
    
    # Render the HTML template with the prediction result
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
