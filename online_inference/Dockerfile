FROM python:3.9

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY x_test.csv /x_test.csv
COPY model.pkl /model.pkl
COPY app_broken.py /app_broken.py

WORKDIR .

CMD ["uvicorn", "app_broken:app_broken", "--host", "0.0.0.0", "--port", "8000"]