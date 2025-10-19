import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Khai báo nhãn (thứ tự phải khớp với lúc train)
labels = ["U thần kinh đệm","U màng não","Không có khối u","U tuyến yên"]

# Load model
model = load_model("models/best_model.keras")

# Đường dẫn thư mục test
test_path = "data/data_split/test"

# Tạo generator cho test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    color_mode='grayscale',
    shuffle=False
)

# Dự đoán toàn bộ test set
y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)  
y_true = test_generator.classes              

# In báo cáo
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=labels, digits=4))

# Vẽ confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()