import tensorflow as tf 
import matplotlib.pyplot as plt


class my_call_back(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.99):
            print("\nReached 99 percent accuracy... \n Stopping training")
            self.model.stop_training = True

callbacks = my_call_back()

(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
#plt.imshow(training_images[0])
#print(training_labels[0])
#plt.show()

print('train_length: ', len(training_images))
print('test_length: ', len(test_images))

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(1024, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer=tf.train.AdamOptimizer(), loss ='sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(training_images,training_labels, epochs=7, callbacks= [callbacks])
model.evaluate(test_images,test_labels)
