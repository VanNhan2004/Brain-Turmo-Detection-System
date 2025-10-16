from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import model và data
from model import model
from data_processing import train_generator, valid_generator

# Biên dịch mô hình
optimizer = Adam(learning_rate=0.0001)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Số epoch
EPOCHS = 100

# Callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath="models/cnn_model.keras",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Huấn luyện
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint],
    verbose=1
)
