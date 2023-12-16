FROM python:3.11-slim

WORKDIR /flask-app

COPY . ./

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "flask_app_capstone.py"]
