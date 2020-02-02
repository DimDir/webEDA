FROM python:3.7

COPY run.py requirements.txt ./

RUN pip install -r requirements.txt

CMD ["streamlit","run","./run.py"]
