Next we're going to replace/create README.md with the proper Hiver assignment documentation.

It will contain:

Project overview
Problem statement
Dataset
Data preparation
AppleSupport conversation extraction
Intent taxonomy
Golden evaluation set
Retrieval architecture
Response generation
FastAPI
Evaluation
Example API request/response
Installation
Running locally
Limitations








python -m venv venv
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn src.api:app --reload

http://127.0.0.1:8000/docs