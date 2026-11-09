# Filo servisi. PyTorch yok: çıkarım ONNX fp32 ile (torch ile farkı 1e-6 mertebesinde, bkz. reports/edge_*.json).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY reports/ reports/
COPY models/ETTh1/latest.json models/ETTh1/*.onnx models/ETTh1/
COPY models/ETTh2/latest.json models/ETTh2/*.onnx models/ETTh2/
# Veri depoda tutulmaz (CC BY-ND 4.0); imaj oluşturulurken kaynağından indirilir.
RUN python -c "from trafo.data import download; download('ETTh1'); download('ETTh2')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
