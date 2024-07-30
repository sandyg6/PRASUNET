import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load the model and label mapping
model = load_model('food_recognition_model.h5')
with open('label_mapping.pkl', 'rb') as f:
    class_indices = pickle.load(f)

# Calorie dictionary
calorie_dict = {
    'apple_pie': 300, 'baby_back_ribs': 400, 'macarons': 160, 'french_toast': 280,
    'lobster_bisque': 250, 'prime_rib': 700, 'pork_chop': 300, 'guacamole': 150,
    'baby_back_ribs': 500, 'mussels': 150, 'beef_carpaccio': 180, 'poutine': 740,
    'hot_and_sour_soup': 100, 'seaweed_salad': 70, 'foie_gras': 450, 'dumplings': 170,
    'peking_duck': 340, 'takoyaki': 350, 'bibimbap': 490, 'falafel': 330,
    'pulled_pork_sandwich': 550, 'lobster_roll_sandwich': 450, 'carrot_cake': 350,
    'beet_salad': 150, 'panna_cotta': 230, 'donuts': 300, 'red_velvet_cake': 400,
    'grilled_cheese_sandwich': 400, 'cannoli': 220, 'spring_rolls': 130,
    'shrimp_and_grits': 400, 'clam_chowder': 250, 'omelette': 150, 'fried_calamari': 150,
    'caprese_salad': 300, 'oysters': 60, 'scallops': 200, 'ramen': 400,
    'grilled_salmon': 350, 'croque_madame': 500, 'filet_mignon': 350, 'hamburger': 250,
    'spaghetti_carbonara': 350, 'miso_soup': 80, 'bread_pudding': 300, 'lasagna': 400,
    'crab_cakes': 350, 'cheesecake': 500, 'spaghetti_bolognese': 450, 'cup_cakes': 150,
    'creme_brulee': 300, 'waffles': 310, 'fish_and_chips': 800, 'paella': 350,
    'macaroni_and_cheese': 350, 'chocolate_mousse': 250, 'ravioli': 350, 'chicken_curry': 300,
    'caesar_salad': 180, 'nachos': 350, 'tiramisu': 450, 'frozen_yogurt': 120,
    'ice_cream': 200, 'risotto': 350, 'club_sandwich': 500, 'strawberry_shortcake': 300,
    'steak': 650, 'churros': 120, 'garlic_bread': 200, 'baklava': 300, 'bruschetta': 120,
    'hummus': 150, 'chicken_wings': 200, 'greek_salad': 150, 'tuna_tartare': 200,
    'chocolate_cake': 350, 'gyoza': 200, 'eggs_benedict': 350, 'deviled_eggs': 70,
    'samosa': 250, 'sushi': 250, 'breakfast_burrito': 300, 'ceviche': 200,
    'beef_tartare': 250, 'apple_pie': 300, 'huevos_rancheros': 350, 'beignets': 250,
    'pizza': 270, 'edamame': 120, 'french_onion_soup': 250, 'hot_dog': 300, 'tacos': 200,
    'chicken_quesadilla': 300, 'pho': 400, 'gnocchi': 200, 'pancakes': 350,
    'fried_rice': 400, 'cheese_plate': 400, 'onion_rings': 400, 'escargots': 100,
    'sashimi': 200, 'pad_thai': 350, 'french_fries': 400
}

# Function to predict and estimate calories
def predict_and_estimate_calories(image_path, model, class_indices, calorie_dict):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0]).numpy()
    label_map = {v: k for k, v in class_indices.items()}
    predicted_label = label_map[predicted_class]

    estimated_calories = calorie_dict[predicted_label]
    
    # Providing diet advice
    if estimated_calories > 500:
        advice = "This is a high-calorie food. Consider balancing with low-calorie options."
    elif estimated_calories > 200:
        advice = "This is a moderate-calorie food. Enjoy in moderation."
    else:
        advice = "This is a low-calorie food. Good choice for a light meal."
    
    return predicted_label, estimated_calories, advice

# Streamlit application
st.title("Food Recognition and Calorie Estimation")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create temp directory if it doesn't exist
    if not os.path.exists('./temp'):
        os.makedirs('./temp')

    img_path = f"./temp/{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(img_path, caption="Uploaded Image", use_column_width=True)
    
    label, calories, advice = predict_and_estimate_calories(img_path, model, class_indices, calorie_dict)
    
    st.write(f"**Predicted Food Item:** {label}")
    st.write(f"**Estimated Calories:** {calories} kcal")
    st.write(f"**Diet Advice:** {advice}")
