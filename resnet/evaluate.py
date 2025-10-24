import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_processing import test_generator
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Khai báo nhãn (thứ tự phải khớp với lúc train)
labels = ["U thần kinh đệm","U màng não","Không có khối u","U tuyến yên"]

# Load model
model = load_model("models/resnet_model.keras")

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


# Load lại history
with open("history/history_resnet.pkl", "rb") as f:
    history = pickle.load(f)

# Vẽ biểu đồ
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()