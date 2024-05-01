import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist


(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)


generator = Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28, 1))
])


discriminator = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])


discriminator.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')


discriminator.trainable = False


gan = Sequential([generator, discriminator])
gan.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')


def train_gan(epochs, batch_size):
    for epoch in range(epochs):
        for _ in range(batch_size):
 
            real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
            
        
            noise = np.random.normal(0, 1, (batch_size, 100))
            
        
            generated_images = generator.predict(noise)
            
       
            x_combined = np.concatenate([real_images, generated_images])
            y_combined = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            discriminator_loss = discriminator.train_on_batch(x_combined, y_combined)
            
        
            noise = np.random.normal(0, 1, (batch_size, 100))
            gan_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
            
      
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Discriminator Loss: {discriminator_loss}, GAN Loss: {gan_loss}")


train_gan(epochs=5000, batch_size=32)



def generate_art_image():
    noise = np.random.normal(0, 1, (1, 100))
    generated_image = generator.predict(noise)[0, :, :, 0]
    plt.imshow(generated_image, cmap='gray')
    plt.axis('off')
    plt.imsave('generated_image.png', generated_image, cmap='gray')  # Salvar a imagem
    plt.show()



generate_art_image()
