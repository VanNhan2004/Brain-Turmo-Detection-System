from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Flatten

model = Sequential()

# CNN layer 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

# CNN layer 2
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# CNN layer 3
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Chuyển đổi dữ liệu đầu ra thành mảng 1 chiều
model.add(Flatten())

# Layer ẩn
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Layer đầu ra
model.add(Dense(4, activation='softmax')) # 4 đầu ra

