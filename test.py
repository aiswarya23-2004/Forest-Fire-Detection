import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# load trained model
model = load_model("model-saves/Inception_based/best_trained_save.h5")

# class labels (same order used in training)
classes = ["fire", "no_fire", "start_fire"]

# ask user for image file
img_path = input("Enter image name (example: fire.jpg): ")

# load image
img = image.load_img(img_path, target_size=(224,224))

# convert image to array
img = image.img_to_array(img)

# apply preprocessing used during training
img = preprocess_input(img)

# add batch dimension
img = np.expand_dims(img, axis=0)

# predict
prediction = model.predict(img)

print("\nClass probabilities:")

for i in range(len(classes)):
    print(classes[i], ":", round(prediction[0][i]*100,2), "%")

# final predicted class
predicted_class = classes[np.argmax(prediction)]
confidence = np.max(prediction)*100

print("\nFinal Prediction:", predicted_class)
print("Confidence:", round(confidence,2), "%")