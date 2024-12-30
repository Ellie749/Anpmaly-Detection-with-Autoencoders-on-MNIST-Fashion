from keras import Model
from keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Input, Layer


class Autoencoder(Layer):
    def __init__(self, n_classes, image_input_shape):
        super().__init__()
        self.n_classes = n_classes
        self.image_input_shape = image_input_shape
        
        # Encoder
        self.e_c1 = Conv2D(4, (3,3), activation='relu', padding='same')
        self.e_c2 = Conv2D(8, (3,3), activation='relu', padding='same')
        self.e_mp1 = MaxPooling2D((2,2))
        self.e_c3 = Conv2D(16, (3,3), activation='relu', padding='same')
        self.e_c4 = Conv2D(32, (3,3), activation='relu', padding='same')
        self.e_mp2 = MaxPooling2D((2,2))

        # Decoder
        self.d_c4 = Conv2DTranspose(32, (3,3), strides=(2,2), activation='relu', padding='same')
        self.d_c3 = Conv2D(16, (3,3), activation='relu', padding='same')
        self.d_c2 = Conv2DTranspose(8, (3,3), strides=(2,2), activation='relu', padding='same')
        self.d_c1 = Conv2D(4, (3,3), activation='relu', padding='same')
        self.mask = Conv2D(3, (1,1), activation='sigmoid', padding='same')
        

    def call(self, x):
        e = self.e_c1(x)
        e = self.e_c2(e)
        e = self.e_mp1(e)
        e = self.e_c3(e)
        e = self.e_c4(e)
        e = self.e_mp2(e)
        d = self.d_c4(e)
        d = self.d_c3(d)
        d = self.d_c2(d)
        d = self.d_c1(d)
        out = self.mask(d)

        return out


    def build(self):
        inputs = Input(shape=self.image_input_shape)
        outputs = self.call(inputs)
        return Model(inputs, outputs)
