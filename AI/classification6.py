import os
import shutil

# 폴더 지우기
shutil.rmtree('/content/dataset')

import os

# 폴더 만들기
os.mkdir('/content/dataset')
os.mkdir('/content/dataset/top')
os.mkdir('/content/dataset/top_blouse')
os.mkdir('/content/dataset/bottom_pants')
os.mkdir('/content/dataset/bottom_skirt')
os.mkdir('/content/dataset/onepiece')
os.mkdir('/content/dataset/onepiece_jumpsuit')
os.mkdir('/content/dataset/outer')
os.mkdir('/content/dataset/outer_padding')


# zip 압축 풀곳 지정하면서 풀기

# 하의
%cd /content/dataset/bottom_pants/
!unzip -qq "/content/drive/MyDrive/[졸프]/AI/학습 데이터/스커트 제외 하의/스커트 제외 하의.zip"
%cd /content/dataset/bottom_skirt/
!unzip -qq "/content/drive/MyDrive/[졸프]/AI/학습 데이터/스커트/스커트.zip"

# 아우터
%cd /content/dataset/outer_padding/
!unzip -qq "/content/drive/MyDrive/[졸프]/AI/학습 데이터/패딩/패딩.zip의 사본"
%cd /content/dataset/outer/
!unzip -qq "/content/drive/MyDrive/[졸프]/AI/학습 데이터/아우터/코트.zip의 사본"
!unzip -qq "/content/drive/MyDrive/[졸프]/AI/학습 데이터/아우터/자켓.zip의 사본"
!unzip -qq "/content/drive/MyDrive/[졸프]/AI/학습 데이터/아우터/가디건.zip의 사본"

# 원피스
%cd /content/dataset/onepiece_jumpsuit/
!unzip -qq "/content/drive/MyDrive/[졸프]/AI/학습 데이터/점프수트/점프수트.zip의 사본"
%cd /content/dataset/onepiece/
!unzip -qq "/content/drive/MyDrive/[졸프]/AI/학습 데이터/원피스/원피스.zip의 사본"
!unzip -qq "/content/drive/MyDrive/[졸프]/AI/학습 데이터/원피스/숏 원피스.zip의 사본"

# 상의
%cd /content/dataset/top_blouse/
!unzip -qq "/content/drive/MyDrive/[졸프]/AI/학습 데이터/블라우스/반팔블라우스.zip의 사본"
!unzip -qq "/content/drive/MyDrive/[졸프]/AI/학습 데이터/블라우스/긴팔블라우스.zip의 사본"
%cd /content/dataset/top/
!unzip -qq "/content/drive/MyDrive/[졸프]/AI/학습 데이터/상의/후드티.zip의 사본"
!unzip -qq "/content/drive/MyDrive/[졸프]/AI/학습 데이터/상의/졸업프로젝트_손찬영.zip의 사본"


import tensorflow as tf

# tf.keras로 이미지 숫자화 하기
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/dataset/', # 이미지가 있는 경로
    # image_size=(64, 64), # 이미지 사이즈 전처리
    image_size=(150, 150),
    batch_size=64, # 64개씩 뽑아 w값 계산, 갱신
    subset='training', # trainig dataset 이름
    validation_split=0.2, # 20%를 validation dataset으로 사용
    seed=1234
)
# validation split을 하고 싶다면 꼭 이것까지 써줘야 되는 것임
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/dataset/', # 이미지가 있는 경로
    # image_size=(64, 64), # 이미지 사이즈 전처리
    image_size=(150, 150),
    batch_size=64, # 64개씩 뽑아 w값 계산, 갱신
    subset='validation', # trainig dataset 이름
    validation_split= 0.2, # 20%를 validation dataset으로 사용
    seed=1234
)

print(train_ds) # 데이터에 대해 출력(위에서 seed를 써줘야함)

#0~255를 255로 나눠 0~1로 압축(선택사항)
def 전처리함수(i, 정답):
  i = tf.cast( i/255.0, tf.float32 )
  return i, 정답

train_ds = train_ds.map(전처리함수)
val_ds = val_ds.map(전처리함수)



# 패션 데이터와 Mnist 개고양이 섞어서 편집
import tensorflow as tf

class_names = ['Top', 'Top_Blouse', 'Bottom_Pants', 'Bottom_Skirt', 'Onepiece', 'Onepiece_Jumpsuit', 'Outer', 'Outer_Padding']

model = tf.keras.Sequential([
                             
  # 이미지 증강
  # tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(64,64,3) ),
  # tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
  # tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),

  tf.keras.layers.Conv2D( 32, (3,3), padding="same", activation='relu', input_shape=(64,64,3) ),
  tf.keras.layers.MaxPooling2D( (2,2) ),
  tf.keras.layers.Conv2D( 64, (3,3), padding="same", activation='relu' ),
  tf.keras.layers.MaxPooling2D( (2,2) ),
  tf.keras.layers.Dropout(0.2), # 노드중 20% 제거(가장 쉬운 오버피팅 해결 방법)
  tf.keras.layers.Conv2D( 128, (3,3), padding="same", activation='relu' ),
  tf.keras.layers.MaxPooling2D( (2,2) ),
  tf.keras.layers.Flatten(), # 데이터를 일렬로 나열
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(8, activation="softmax"),
])

model.summary()

model.compile( loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'] )
model.fit(train_ds, validation_data=val_ds, epochs=5)


# inception_v3.h5 파일 다운받기
import os

from tensorflow.keras import layers
from tensorflow.keras import Model

!wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 -O inception_v3.h5



# InceptionV3 모델 불러오기
from tensorflow.keras.applications.inception_v3 import InceptionV3

inception_model = InceptionV3( input_shape=(150,150,3), include_top=False, weights=None )
inception_model.load_weights('inception_v3.h5')

inception_model.summary()

for i in inception_model.layers:
  i.trainable = False

# fine tuning 할 때
# unfreeze = False
# for i in inception_model.layers:
#   if i.name == 'mixed6':
#     unfreeze = True
#   if unfreeze == True:
#     i.trainable = True

마지막레이어 = inception_model.get_layer('mixed7')

print(마지막레이어)
print(마지막레이어.output)
print(마지막레이어.output_shape)


import tensorflow as tf

layer1 = tf.keras.layers.Flatten()(마지막레이어.output)
layer2 = tf.keras.layers.Dense(1024, activation='relu')(layer1)
drop1 = tf.keras.layers.Dropout(0.2)(layer2)
layer3 = tf.keras.layers.Dense(8, activation='softmax')(drop1)

model = tf.keras.Model(inception_model.input, layer3)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(train_ds, validation_data=val_ds, epochs=2)



# fine tuning 할 때

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.00001), metrics=['acc'])
model.fit(train_ds, validation_data=val_ds, epochs=2)
