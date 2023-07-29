# import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.keras import Model
import tensorflow_datasets as tfds


class MovieReviewClassificationModel(Model):
  
    def __init__(self):
        super(MovieReviewClassificationModel, self).__init__()

        self._load_dataset()
        self._build_model()
        # self._print_summary()


    @tf.function
    def call(self, inputs):

        print("-------------------------- Calling the model")
        out = self.model(inputs)
        return out
        
    def compile_model(self):

        print("-------------------------- Compiling the model")
        self.model.compile(optimizer='adam',
            loss=tf.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])



    def _build_model(self):

        print("-------------------------- Building the model")
        self.model = tf.keras.models.Sequential()
        self.model.add(self._create_first_layer())
        self.model.add(tf.keras.layers.Dense(16, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1))

    def _print_summary(self):

        print("-------------------------- Model summary:")
        self.model.summary()


    def _load_dataset(self):

        print("-------------------------- Loading dataset")
        train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], 
                                          batch_size=-1, as_supervised=True)

        self.train_examples, self.train_labels = tfds.as_numpy(train_data)
        self.test_examples, self.test_labels = tfds.as_numpy(test_data)

        print("Training entries: {}, test entries: {}".format(len(self.train_examples), len(self.test_examples)))

    def load_weights(self, ckpt_dir):
        self.model.load_weights(ckpt_dir).expect_partial()


    def _create_first_layer(self):

        print("-------------------------- Creating the layer")
        model_name = "https://tfhub.dev/google/nnlm-en-dim50/2"
        hub_layer = hub.KerasLayer(model_name, input_shape=[], dtype=tf.string, trainable=True)
        # hub_layer(train_examples[:3])
        return hub_layer
    
    def predict(self, reviewString: str):
        print ("-------------------------- Predicting")
        return self.model.predict([reviewString]).tolist()[0][0]

    def evaluate(self):

        print("-------------------------- Evaluating the model")
        return self.model.evaluate(self.test_examples, self.test_labels)
    

    def train(self, ckpt_dir):
        callbacks = []
        cp_callback = tf.keras.callbacks.ModelCheckpoint( filepath=ckpt_dir,
                                                            save_weights_only=True,
                                                            verbose=1 )
        callbacks.append(cp_callback)

        print("-------------------------- Fitting the model")

        x_val = self.train_examples[:10000]
        partial_x_train = self.train_examples[10000:]

        y_val = self.train_labels[:10000]
        partial_y_train = self.train_labels[10000:]

        self.history = self.model.fit(  partial_x_train,
                                        partial_y_train,
                                        epochs=20,
                                        batch_size=2048,
                                        validation_data=(x_val, y_val),
                                        verbose=1,
                                        callbacks=callbacks,)

        self.model.save_weights(ckpt_dir)