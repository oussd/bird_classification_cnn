from django.shortcuts import render
import requests
from train.model import build_model

def home(request):
    if request.method == 'POST' and request.FILES['image']:
        file = request.FILES['image']
        if file:
            # Save the uploaded image locally
            with open('uploaded_image.jpg', 'wb') as f:
                for chunk in file.chunks():
                    f.write(chunk)
            
            # Send the image to the model for prediction
            url = 'http://model_container_ip:5000/predict'  # Replace with the IP address of your model container
            files = {'file': open('uploaded_image.jpg', 'rb')}
            response = requests.post(url, files=files)
            
            # Get the prediction from the model
            prediction = response.json()['prediction']
            
            # Pass prediction to the template
            return render(request, 'result.html', {'prediction': prediction})
    
    return render(request, 'upload.html')
