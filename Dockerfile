FROM python:3.11.1

WORKDIR /VE

EXPOSE 8501

COPY . /VE

RUN pip install -r requirement.txt

CMD streamlit run server.py