# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
df_twitter= pd.read_csv('twitter_data.csv', header=None, names=[f'col_{i}' for i in range(4)])

# %%
print(df_twitter.info())

# %%
df_twitter.dropna(subset=['col_3'], inplace=True)

# %%
print(df_twitter.info())

# %%
# Split dataset Twitter
text_twitter = df_twitter['col_3'].values 
label_twitter = df_twitter[['col_2']].values 

text_train_twitter, text_test_twitter, label_train_twitter, label_test_twitter = train_test_split(
    text_twitter, label_twitter, test_size=0.2, random_state=42, shuffle=True
)

# %%
model_twitter = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=15000, output_dim=64),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])

model_twitter.summary()

# %%
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.9 and logs.get('val_accuracy') > 0.9:
            self.model.stop_training = True
            print("\nThe accuracy of the training set and the validation set has reached > 90%!")

callbacks_twitter = MyCallback()

# %%
# Handling NaN values in the text column
text_train_twitter = text_train_twitter.astype(str)
text_test_twitter = text_test_twitter.astype(str)

# Tokenizing, Sequencing, dan Padding
tokenizer_twitter = Tokenizer(num_words=15000, oov_token='<oov>', filters='!"#$%&()*+,-./:;<=>@[\\]^_`{|}~ ')
tokenizer_twitter.fit_on_texts(text_train_twitter)
tokenizer_twitter.fit_on_texts(text_test_twitter)

sequences_train_twitter = tokenizer_twitter.texts_to_sequences(text_train_twitter)
sequences_test_twitter = tokenizer_twitter.texts_to_sequences(text_test_twitter)

padded_train_twitter = pad_sequences(sequences_train_twitter)
padded_test_twitter = pad_sequences(sequences_test_twitter)

# %%
from sklearn.preprocessing import LabelEncoder

# %%
label_encoder = LabelEncoder()
label_train_twitter_encoded = label_encoder.fit_transform(label_train_twitter)
label_test_twitter_encoded = label_encoder.transform(label_test_twitter)

# %%
num_classes = len(label_encoder.classes_)

# %%
from tensorflow.keras.utils import to_categorical

# %%
label_train_twitter_onehot = to_categorical(label_train_twitter_encoded, num_classes=num_classes)  
label_test_twitter_onehot = to_categorical(label_test_twitter_encoded, num_classes=num_classes)

# %%
# Compile model Twitter
model_twitter.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')  # Sesuaikan dengan loss dan jenis label


# %%
# Training model
history_twitter = model_twitter.fit(padded_train_twitter, label_train_twitter_onehot, epochs=50,
                                    validation_data=(padded_test_twitter, label_test_twitter_onehot), verbose=2, callbacks=[callbacks_twitter])

# %%
# Evaluasi model Twitter
model_twitter.evaluate(padded_test_twitter, label_test_twitter_onehot)

# %%
# Visualisasi hasil training model Twitter
loss_twitter = history_twitter.history['loss']
val_loss_twitter = history_twitter.history['val_loss']

acc_twitter = history_twitter.history['accuracy']
val_acc_twitter = history_twitter.history['val_accuracy']

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss_twitter, label='Training set')
plt.plot(val_loss_twitter, label='Validation set', linestyle='--')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)

plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(acc_twitter, label='Training set')
plt.plot(val_acc_twitter, label='Validation set', linestyle='--')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)

plt.show()



