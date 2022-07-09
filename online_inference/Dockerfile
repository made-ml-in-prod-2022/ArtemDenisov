FROM python:3.9

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY x_test.csv /x_test.csv
COPY model.pkl /model.pkl
COPY app.py /app.py

WORKDIR .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]