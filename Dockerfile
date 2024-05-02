# FROM tiangolo/uwsgi-nginx-flask

# # LABEL image for a very simple flask application
# COPY ./requirements.txt /docker-flask/requirements.txt
# ENV GOOGLE_APPLICATION_CREDENTIALS /key.json
# WORKDIR /docker-flask
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
# COPY . .
# EXPOSE 8080
# CMD [ "python","main.py" ]

FROM python


# We copy just the requirements.txt first to leverage Docker cache

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python" ]

CMD [ "main.py" ]