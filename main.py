import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import pdb

def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    return (train_images, train_labels), (test_images, test_labels)

def create_model():
    pdb.set_trace()  # Breakpoint for debugging model creation
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def compile_and_train_model(model, train_images, train_labels):
    pdb.set_trace()  # Breakpoint for debugging model compilation and training
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    return model

def evaluate_model(model, test_images, test_labels):
    pdb.set_trace()  # Breakpoint for debugging model evaluation
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')

if _name_ == "_main_":
    (train_images, train_labels), (test_images, test_labels) = load_data()
    model = create_model()
    model = compile_and_train_model(model, train_images, train_labels)
    evaluate_model(model, test_images, test_labels)
    model.save('mnist_cnn_model.h5')  # Save the model
