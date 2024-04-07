docker build -t django_app -f django_app.dockerfile .

git checkout train
cp -r ./models/bird_classification_model.h5 ./model


docker run -p 8000:8000 django_app

