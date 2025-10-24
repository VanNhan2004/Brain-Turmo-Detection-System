from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_resnet import model
from data_processing import train_generator, valid_generator
import pickle

# Biên dịch mô hình
optimizer = Adam(learning_rate=0.0001)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath="models/resnet_model.keras",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Huấn luyện
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs = 100,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

with open("history/history_resnet.pkl", 'wb') as f:
    pickle.dump(history.history, f)