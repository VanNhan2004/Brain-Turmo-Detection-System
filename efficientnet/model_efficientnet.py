from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Conv2D
from tensorflow.keras.models import Model

inputs = Input(shape=(224, 224, 1))
x = Conv2D(3, (3,3), padding='same')(inputs)   

base_model = EfficientNetB0(
    weights=None,              
    include_top=False,
    input_tensor=x
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(4, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)


