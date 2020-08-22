'''
1. import modules
'''
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, BatchNormalization
import tensorflow as tf 
np.random.seed(1234)
tf.random.set_seed(1234)


'''
2. load data
'''
(x_train,y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train=(x_train.reshape(-1,784)/255).astype(np.float32)
x_test=(x_test.reshape(-1,784)/255).astype(np.float32)
y_train = np.eye(10)[y_train].astype(np.float32)
y_test = np.eye(10)[y_test].astype(np.float32)


'''
3. build a model
'''
# method 1
model = Sequential([
    Dense(200,activation='relu',input_shape=(784,)),
    Dense(10,activation='softmax')
])
model.summary()
# method 2
input = Input(shape=(784,))
x=Dense(200,activation='relu')(input)
output=Dense(10,activation='softmax')(x)
model = Model(inputs=input, outputs=output)
model.summary()
# method 3
class MLP(Model):
    def __init__(self):
        super().__init__()
        self.dense=Dense(200,activation='relu')
        self.out=Dense(10,activation='softmax')
    def call(self,x):
        x=self.dense(x)
        y=self.out(x)
        return y
model = MLP()
# note: use (None,784) rather (784,) which is used in the layer (check method 1)
model.build(input_shape=(None,784)) 
model.summary()


'''
4. compile a model
'''
criterion = tf.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,loss=criterion,metrics=['accuracy'])


'''
5. train and evaluate a model
'''
# method1 use built-in functions
model.fit(x_train,y_train,epochs=2,batch_size=100)
loss, accuracy =model.evaluate(x_test,y_test)
print("loss is {}, accuracy is {}".format(loss,accuracy))

# method2 write customized loop
# define some TF functions
@tf.function
def compute_loss(label, pred):
    return criterion(label, pred)

@tf.function
def train_step(x, t):
    with tf.GradientTape() as tape:
        preds = model(x)
        loss = compute_loss(t, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)
    train_acc(t, preds)

    return preds

@tf.function
def test_step(x, t):
    preds = model(x)
    loss = compute_loss(t, preds)
    test_loss(loss)
    test_acc(t, preds)

    return preds

epochs = 2
batch_size = 100
n_batches = x_train.shape[0] // batch_size

train_loss = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.CategoricalAccuracy()
test_loss = tf.keras.metrics.Mean()
test_acc = tf.keras.metrics.CategoricalAccuracy()

for epoch in range(epochs):

    from sklearn.utils import shuffle
    _x_train, _y_train = shuffle(x_train, y_train, random_state=42)

    for batch in range(n_batches):
        start = batch * batch_size
        end = start + batch_size
        train_step(_x_train[start:end], _y_train[start:end])

    if epoch % 5 == 4 or epoch == epochs - 1:
        preds = test_step(x_test, y_test)
        print('Epoch: {}, Valid Cost: {:.3f}, Valid Acc: {:.3f}'.format(
            epoch+1,
            test_loss.result(),
            test_acc.result()
        ))