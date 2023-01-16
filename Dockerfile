FROM python:3.10

RUN pip3 install --upgrade pip

COPY ./laugmentateur ./laugmentateur

WORKDIR ./laugmentateur

EXPOSE 8501

RUN pip3 install -r requirements.txt

CMD streamlit run augment.py


