# multivariate output stacked lstm example
import keras 
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import Adam, SGD
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/code/geneaDatasetCreation')
import bvhLoader
from lstmDataset import lstmDataset
# import plotly.express as px
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from pickle import dump, load
from keras import backend as K
# choose a number of time steps (sequence size)
n_steps = 60
n_steps_out = 1
n_features = 192
learning_rate = 0.001

class customCallback(keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        layer = self.model.layers[0]
        print(layer)
        weights = layer.get_weights()
        print(weights)
    def on_train_batch_end(self, batch, logs=None):
        # demonstrate prediction
        # scaler = load(open("scaler.pkl", "rb"))
        x_input, y, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3sec", partition="All", verbose=True, specificSize=30, trim=True, sequenceSize=n_steps, outSequenceSize=n_steps_out, removeHandsAndFace=True)
        #########################
        # predict the positions #
        #########################
        x_input = x_input[3]
        x_input = np.array(x_input)
        x_input = x_input.reshape((1, n_steps, n_features))
        finalOutput = []
        # prediction loop
        newX = self.model.predict(x_input, verbose=0)
        x_input = x_input.reshape((n_steps, n_features))
        newX = newX.reshape((n_steps_out, n_features))
        finalOutput = np.append(x_input, newX, axis=0)
        # finalOutput = scaler.inverse_transform(finalOutput)
        print(finalOutput.shape)
        with open("testBvh.bvh", "w") as f:
            for line in finalOutput:
                f.write(str(line.tolist()).replace("[", "").replace("]", "").replace(",", ""))
                f.write("\n")
            f.close

def train():
    # create the standard scaler, then pass it to the datamodule
    # scaler = bvhLoader.createAndFitStandardScaler(datasetName = "silenceDataset3sec", removeHandsAndFace=True)
    # dump(scaler, open("scaler.pkl", "wb"))
    scaler = load(open("scaler.pkl", "rb"))
    # convert into input/output
    datamodule = lstmDataset(root="/home/bee/Desktop/idle animation generator", isTiny = False, batchSize = 256, partition="Train", datasetName = "silenceDataset3sec", 
                             sequenceSize = n_steps, trim=False, verbose=True, outSequenceSize=n_steps_out, removeHandsAndFace = True, scaler = scaler)
    datamoduleVal = lstmDataset(root="/home/bee/Desktop/idle animation generator", isTiny = False, batchSize = 256, partition="Validation", datasetName = "silenceDataset3sec", 
                             sequenceSize = n_steps, trim=False, verbose=True, outSequenceSize=n_steps_out, removeHandsAndFace = True, scaler = scaler)
    n_features = len(datamodule.sequences[0][0]) # datamodule.sequences[0][0] is a vector, of dimension n_features
    ####################
    # DEFINE THE MODEL #
    ####################
    # Model that doesn't use the time distributed layer (generates 1 output vector). Test it with the test() function
    # model = Sequential()
    # model.add(LSTM(200, activation = 'relu', input_shape=(n_steps, n_features), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4))
    # model.add(LSTM(200, activation = 'relu', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4))
    # model.add((Dense(n_features)))
    # opt = Adam(learning_rate=learning_rate)
    # model.compile(optimizer=opt, loss='cosine_similarity')
    # model.summary()

    # apply early stopping to the network training
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 8, verbose = 1)

    # Model that uses the time distributed output (generates n_steps_out output vectors). Test it with the testMultiple() function
    model = Sequential()
    model.add(LSTM(200, activation = 'sigmoid', input_shape=(n_steps, n_features), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4, recurrent_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2'))
    # model.add(RepeatVector(n_steps_out))
    model.add(LSTM(200, activation = 'sigmoid', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4, recurrent_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2'))
    model.add(Dense(n_features))
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mae')
    model.summary()

    # fit model
    history = model.fit(datamodule, validation_data=datamoduleVal, epochs=100, verbose=1, callbacks=[early_stopping_callback])

    model.save("models/multipleLSTMNoLimbsEsOneFrame.keras")
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig("resultImages/Training multipleLSTMNoLimbsEsOneFrame.png")
    plt.close()
    # Model that uses the time distributed output (generates n_steps_out output vectors). Test it with the testMultiple() function
    model2 = Sequential()
    model2.add(LSTM(200, activation = 'sigmoid', input_shape=(n_steps, n_features), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4, recurrent_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2'))
    # model2.add(RepeatVector(n_steps_out))
    model2.add(LSTM(200, activation = 'sigmoid', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4, recurrent_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2'))
    model2.add(Dense(n_features))
    opt = Adam(learning_rate=learning_rate)
    model2.compile(optimizer=opt, loss='mse')
    model2.summary()

    # fit model2
    history = model2.fit(datamodule, validation_data=datamoduleVal, epochs=100, verbose=1, callbacks=[early_stopping_callback])

    model2.save("models/multipleLSTMNoLimbsMseOneFrame.keras")
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig("resultImages/Training multipleLSTMNoLimbsMseOneFrame.png")
    plt.close()
    # Model that uses the time distributed output (generates n_steps_out output vectors). Test it with the testMultiple() function
    model3 = Sequential()
    model3.add(LSTM(200, activation = 'sigmoid', input_shape=(n_steps, n_features), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4, recurrent_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2'))
    # model3.add(RepeatVector(n_steps_out))
    model3.add(LSTM(200, activation = 'sigmoid', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4, recurrent_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2'))
    model3.add(Dense(n_features))
    opt = Adam(learning_rate=learning_rate)
    model3.compile(optimizer=opt, loss='cosine_similarity')
    model3.summary()

    # fit model3
    history = model3.fit(datamodule, validation_data=datamoduleVal, epochs=100, verbose=1, callbacks=[early_stopping_callback])

    model3.save("models/multipleLSTMNoLimbsCosineOneFrame.keras")
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig("resultImages/Training multipleLSTMNoLimbsCosineOneFrame.png")
    plt.close()
    # Model that uses the time distributed output (generates n_steps_out output vectors). Test it with the testMultiple() function
    model4 = Sequential()
    model4.add(LSTM(200, activation = 'sigmoid', input_shape=(n_steps, n_features), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4, recurrent_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2'))
    # model4.add(RepeatVector(n_steps_out))
    model4.add(LSTM(200, activation = 'sigmoid', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4, recurrent_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2'))
    model4.add(Dense(n_features))
    opt = Adam(learning_rate=learning_rate)
    model4.compile(optimizer=opt, loss='kl_divergence')
    model4.summary()

    # fit model4
    history = model4.fit(datamodule, validation_data=datamoduleVal, epochs=100, verbose=1, callbacks=[early_stopping_callback])

    model4.save("models/multipleLSTMNoLimbsKldOneFrame.keras")
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig("resultImages/Training multipleLSTMNoLimbsKldOneFrame.png")
    plt.close()
    # Model that uses the time distributed output (generates n_steps_out output vectors). Test it with the testMultiple() function
    model5 = Sequential()
    model5.add(LSTM(200, activation = 'sigmoid', input_shape=(n_steps, n_features), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4, recurrent_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2'))
    # model5.add(RepeatVector(n_steps_out))
    model5.add(LSTM(200, activation = 'sigmoid', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4, recurrent_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2'))
    model5.add(Dense(n_features))
    opt = Adam(learning_rate=learning_rate)
    model5.compile(optimizer=opt, loss='mape')
    model5.summary()

    # fit model5
    history = model5.fit(datamodule, validation_data=datamoduleVal, epochs=100, verbose=1, callbacks=[early_stopping_callback])

    model5.save("models/multipleLSTMNoLimbsMapeOneFrame.keras")
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig("resultImages/Training multipleLSTMNoLimbsMapeOneFrame.png")
    plt.close()
    # Model that uses the time distributed output (generates n_steps_out output vectors). Test it with the testMultiple() function
    model6 = Sequential()
    model6.add(LSTM(200, activation = 'sigmoid', input_shape=(n_steps, n_features), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4, recurrent_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2'))
    # model6.add(RepeatVector(n_steps_out))
    model6.add(LSTM(200, activation = 'sigmoid', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4, recurrent_regularizer='l2', bias_regularizer='l2', activity_regularizer='l2'))
    model6.add(Dense(n_features))
    opt = Adam(learning_rate=learning_rate)
    model6.compile(optimizer=opt, loss='msle')
    model6.summary()

    # fit model6
    history = model6.fit(datamodule, validation_data=datamoduleVal, epochs=100, verbose=1, callbacks=[early_stopping_callback])

    model6.save("models/multipleLSTMNoLimbsMsleOneFrame.keras")
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig("resultImages/Training multipleLSTMNoLimbsMsleOneFrame.png")
    plt.close()
def test(modelName):
    # demonstrate prediction
    model = load_model(modelName) # "models/singleLSTMNoLimbsOneFrame.keras"
    scaler = load(open("scaler.pkl", "rb"))
    x_input, y, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3sec", partition="Validation", verbose=True, specificSize=10, trim=True, sequenceSize=n_steps, removeHandsAndFace=True, scaler=scaler)
    x_input = x_input[0]
    x_input = x_input.reshape((1, n_steps, n_features))
    newX = model.predict(x_input, verbose=0)
    finalOutput = []
    finalOutput = np.append(x_input[0], newX, axis=0)
    # prediction loop
    steps = 30
    for step in range(0, steps):
        x_input = x_input.reshape(n_steps, n_features)
        x_input = np.append(x_input, newX, axis=0)
        x_input = np.delete(x_input, 0, axis=0)
        x_input = x_input.reshape((1, n_steps, n_features))
        newX = model.predict(x_input, verbose=0)
        finalOutput = np.append(finalOutput, newX, axis=0)

    finalOutput = scaler.inverse_transform(finalOutput)

    with open("resultBvhs/" + modelName.split(".")[0].split("/")[1] + ".bvh", "w") as f:
        for line in finalOutput:
            f.write(str(line.tolist()).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")
        f.write(str(newX.tolist()).replace("[", "").replace("]", "").replace(",", ""))
        f.close

def testMultiple(modelName):
    # demonstrate prediction
    model = load_model(modelName) # "models/multipleLSTMNoLimbsOneFrame.keras"
    scaler = load(open("scaler.pkl", "rb"))
    x_input, y, ids = bvhLoader.loadSequenceDataset(datasetName="silenceDataset3sec", partition="Validation", verbose=True, specificSize=200, trim=True, sequenceSize=n_steps, outSequenceSize=n_steps_out, removeHandsAndFace=True, scaler = scaler)
    #########################
    # predict the positions #
    #########################
    x_input = x_input[0]
    x_input = np.array(x_input)
    x_input = x_input.reshape((1, n_steps, n_features))
    finalOutput = []
    # prediction loop
    newX = model.predict(x_input, verbose=0)
    x_input = x_input.reshape((n_steps, n_features))
    newX = newX.reshape((n_steps_out, n_features))
    finalOutput = np.append(x_input, newX, axis=0)
    finalOutput = scaler.inverse_transform(finalOutput)
    print(finalOutput.shape)
    with open("resultBvhs/" + modelName.split(".")[0].split("/")[1] + ".bvh", "w") as f:
        for line in finalOutput:
            f.write(str(line.tolist()).replace("[", "").replace("]", "").replace(",", ""))
            f.write("\n")
        f.close

def checkDatamodule():
    # convert into input/output
    datamodule = lstmDataset(root="/home/bee/Desktop/idle animation generator", isTiny = False, batchSize= 56, partition="All", datasetName = "silenceDataset3sec", 
                             sequenceSize = n_steps, trim=False, specificSize=10, verbose=True, outSequenceSize=n_steps_out, removeHandsAndFace = True)
    
    print(len(datamodule[0][0][0]))
    print(len(datamodule[0][1][0]))

def checkNetwork():
    model = Sequential()
    model.add(LSTM(200, activation = 'relu', input_shape=(n_steps, n_features), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4))
    # model.add(RepeatVector(n_steps_out))
    model.add(LSTM(200, activation = 'relu', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4))
    model.add((Dense(n_features)))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

def visualizeWeights():
    model = load_model("models/multipleLSTMNoLimbsOneFrame.keras")
    layer = model.layers[0]
    print(layer)
    weights = layer.get_weights()
    print(weights)
    
if __name__ == "__main__":
    train()
    test("models/multipleLSTMNoLimbsEsOneFrame.keras")
    test("models/multipleLSTMNoLimbsCosineOneFrame.keras")
    test("models/multipleLSTMNoLimbsKldOneFrame.keras")
    test("models/multipleLSTMNoLimbsMapeOneFrame.keras")
    test("models/multipleLSTMNoLimbsMsleOneFrame.keras")
    test("models/multipleLSTMNoLimbsMseOneFrame.keras")
    # checkDatamodule()
    # visualizeWeights()
