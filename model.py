import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 랜덤 시드 값을 설정하여 재현 가능성을 확보
tf.random.set_seed(1234)

# 이미지 데이터를 [0,1] 사이의 값으로 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0

# 데이터의 차원을 4D로 변경 (배치 크기, 높이, 너비, 채널 수)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 레이블 데이터를 one-hot 인코딩 형태로 변경
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 이미지 데이터 증강을 위한 설정 회전, 이동, 확대 변환을 적용
datagen = ImageDataGenerator(
    rotation_range=10,      # 10도 범위 내에서 이미지를 무작위로 회전
    width_shift_range=0.1,  # 이미지를 좌우로 10% 범위 내에서 무작위로 이동
    height_shift_range=0.1, # 이미지를 상하로 10% 범위 내에서 무작위로 이동
    zoom_range=0.1          # 이미지를 10% 범위 내에서 무작위로 확대/축소
)
datagen.fit(x_train)

# CNN 모델을 정의
model = tf.keras.Sequential([
    # 첫 번째 Convolution 레이어
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, input_shape=(28,28,1), padding='same', activation='relu'),
    # 두 번째 Convolution 레이어
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding='same', activation='relu'),
    # 첫 번째 Pooling 레이어
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    # 세 번째 Convolution 레이어
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, padding='same', activation='relu'),
    # 네 번째 Convolution 레이어
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=256, padding='valid', activation='relu'),
    # 두 번째 Pooling 레이어
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    # Flatten 레이어로 2D 데이터를 1D로 변경
    tf.keras.layers.Flatten(),
    # 완전 연결 레이어
    tf.keras.layers.Dense(units=512, activation='relu'),
    # 과적합 방지를 위한 Dropout 레이어
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    # 출력 레이어
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# 모델 컴파일 -> 손실 함수, 최적화 알고리즘, 평가 메트릭을 정의
model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.001), metrics=['accuracy'])
# 모델의 구조를 출력
model.summary()
# 모델 훈련 데이터 증강을 사용한 경우 datagen.flow를 사용하고, 그렇지 않은 경우 기본 훈련 데이터를 사용
model.fit(datagen.flow(x_train, y_train, batch_size=100), validation_data=(x_test, y_test), epochs=10)

# 테스트 데이터로 모델의 성능을 평가
result = model.evaluate(x_test, y_test)
# 테스트 데이터에 대한 정확도를 출력
print("최종 예측 성공률(%): ", result[1]*100)
# 모델 저장
model.save('zz_model.h5')
