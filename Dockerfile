FROM python:3.14-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY mlops_pipeline/src/model_deploy.py mlops_pipeline/src/
COPY mlops_pipeline/src/modelo_rf.joblib mlops_pipeline/src/

EXPOSE 8000

CMD ["uvicorn", "model_deploy:app", "--app-dir", "mlops_pipeline/src", "--host", "0.0.0.0", "--port", "8000"]