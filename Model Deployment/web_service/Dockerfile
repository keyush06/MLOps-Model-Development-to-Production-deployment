FROM python:3.12-slim

RUN pip install -U pip

## we manaed our dependencies using pipenv so we have to run that
RUN pip install pipenv 

## the files wil be copied here
WORKDIR /app

# This will download all the packages in the current working directory
COPY [ "Pipfile", "Pipfile.lock", "./" ]

## becuase we are already isloated - we do not need a virtual environment and we can do to the system
RUN pipenv install --system --deploy

COPY [ "predict.py", "lin_reg.bin", "./" ]

## I have used localhost:9090
EXPOSE 9090

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9090", "predict:app" ]

#------Commnads to be used for docker-----------
    # docker build -t ride-duration-prediction-service:v1 .
    # docker run -it --rm -p 9696:9696  ride-duration-prediction-service:v1

#-----------END------------------END-----------