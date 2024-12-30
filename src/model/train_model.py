
def train(model, X_train, y_train, X_test, y_test, epochs, batch_size):

    model.compile(loss='MSE', optimizer='adam', metrics='accuracy')
    H = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

    return H