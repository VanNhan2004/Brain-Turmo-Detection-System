import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Khai báo nhãn (phải đúng thứ tự class khi train)
labels = ["U thần kinh đệm","U màng não","Không có khối u","U tuyến yên"]

# Load model
model = load_model("models/cnn_model.keras")

# Đường dẫn ảnh test
img_path = "data/data_split/test/normal/N_2.jpg"

# Tiền xử lý ảnh
img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
x = np.expand_dims(image.img_to_array(img), axis=0) / 255.0

# Dự đoán
y_predict = model.predict(x)[0]   # lấy mảng xác suất 1 chiều

# In ra ma trận xác suất
print("Xác suất dự đoán cho từng class:")
for i, prob in enumerate(y_predict):
    print(f"{labels[i]}: {prob:.4f}")

# In ra class dự đoán cao nhất và độ tin cậy
pred_idx = np.argmax(y_predict)
print("\nDự đoán:", labels[pred_idx])
print("Độ tin cậy:", f"{y_predict[pred_idx]*100:.2f}%")

