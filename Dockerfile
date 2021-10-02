FROM spellrun/fastai

WORKDIR /usr/src/app
COPY . .

RUN python3 -m pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["app.py"]