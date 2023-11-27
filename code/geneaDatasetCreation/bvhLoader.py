import os
import numpy as np
import torch

###############################################
# part 1: reading a single bvh file to a list #
###############################################

# reads a bvh file and separates the data from the header. Returns the header if needed, else returns only the data in a list. Can also return just the header.
# each row of the list contains a number of joint rotations. It also returns the number of frames loaded
def loadBvhToList(path, returnHeader = False, returnData = True):
    ### HEADER ###
    # read and save the header in a variable
    f = open(path, "r")
    header = ""
    line = f.readline()
    # read the header until the line "Frame Time: 0.0333333"
    while line.split()[0]!= "Frame":
        header += line
        line = f.readline()
    # add the last header line manually
    # header += line.split("\n")[0]
    header += line

    ### DATA ###
    # read all the rotation data to a list
    data = []
    line = f.readline().replace("\n", "")
    counter = 0
    while True:
        data.append(line.split(" ")[:-1])
        line = f.readline().replace("\n", "")
        counter+=1
        if not line: break
    data = [[np.float32(s) for s in sublist] for sublist in data]
    data = np.asanyarray(data)
    if returnHeader and returnData:
        return header, data, counter
    if returnData:
        return data, counter
    return header, counter

#####################################################################
# part 2: loading a dataset partition, in bulk or divided by person #
#####################################################################

# loads the specified bvh dataset, and the partition can also be specified (if trim = true, all sequences are trimmed to the length of the smallest sequence)
# returns: data format is a list of 3 dimensions [n(number of bvh) person][m (depends on the bvh) frame][498 rotation]
def loadDataset(datasetName, partition = "All", specificSize=-1, verbose = False, trim = False, specificTrim = -1):
    allData = []
    allIds = []
    id = 0
    finalTrimSize = 999999999999999
    if partition=="All":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/"
    if partition=="Train":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/genea2023_trn"
    if partition=="Validation":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/genea2023_val"
    for root, dirs, files in os.walk(path):
        for filename in files:
            if verbose:
                print("Loading file: " + str(filename))
            bvhData, bvhSize = loadBvhToList(os.path.join(root, filename))
            # if the trim flag is on, calculate the size of the smallest bvh
            if trim:
                if finalTrimSize > bvhSize:
                    finalTrimSize = bvhSize
            allData.append(bvhData)
            allIds.append(id) # TODO: IDs should not be numbers. Change to one-hot encoding or other
            if specificSize!=-1 and id>=specificSize:
                break
                # return allData, np.asarray(allIds)
            id+=1

    # after loading the entire dataset, if trim is activated, trim all sequences to the smallest size
    if trim:
        if specificTrim > -1 and specificTrim<=finalTrimSize:
            if verbose:
                print("Trimming to size: " + str(specificTrim))
            # trim
            for person in range(0, len(allData)):
                allData[person] = allData[person][0:specificTrim].copy()
        else:
            if verbose:
                print("Trimming to size: " + str(finalTrimSize))
            for person in range(0, len(allData)):
                allData[person] = allData[person][0:finalTrimSize].copy()
    return allData, np.asarray(allIds)

# loads the specified bvh dataset, and the partition can also be specified
# the data is loaded in bulk, no difference between different frames or people, just a list of all the vectors
def loadDatasetInBulk(datasetName, partition = "All", specificSize=-1, verbose = False):
    img_size = (1, 32, 32)
    allData = []
    if partition=="All":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/"
    if partition=="Train":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/genea2023_trn"
    if partition=="Validation":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/genea2023_val"
    counter = 0
    for root, dirs, files in os.walk(path):
        for filename in files:
            if verbose:
                print("Loading file: " + str(filename))
            vectors = loadBvhToList(os.path.join(root, filename))
            # instead of appending lists representing people, append the vectors individually
            for vector in vectors:
                # bektoreak paddingarekin 1024 dimentsioata pasatzeko (32x32 konboluzioak erabilteko eginda behin)
                #######################################
                # vector = torch.from_numpy(vector)
                # vector = torch.cat((vector, torch.zeros(526)), dim=0)
                # vector = torch.reshape(vector, img_size) 
                #######################################
                allData.append(vector)
                if specificSize!=-1 and counter>=specificSize:
                    return allData
                counter+=1
    return allData

############################################################
# part 3: specific methods to laod different types of data #
############################################################

# creates the second column of the dataset, for each frame, its result (the next frame)
def createResultsFromDataset(dataset):
    newDataset = []
    personResults = []
    for person in dataset:
        for frame in range(0, len(person)-1):
            personResults.append(person[frame+1])
        newDataset.append(personResults.copy())
        personResults.clear()
    return newDataset

# deletes the last row from the dataset, as it does not have an answer (it has no next frame)
def trimLastFrameFromDataset(dataset):
    newDataset = dataset
    for person in newDataset:
        person = np.delete(person, len(person))
    return newDataset

# loads the vectors, and returns the differences between a vector and its next vector
def loadDifferencesDataset(datasetName, partition = "All", specificSize=-1, verbose = False):
    datasetX = loadDatasetInBulk(datasetName=datasetName, partition=partition, specificSize=specificSize, verbose=verbose)
    differencesDataset = []
    for index in range(0, len(datasetX)-1):
        differencesDataset.append(datasetX[index]-datasetX[index+1])
    return differencesDataset

# creates the sequential part of the dataset, and also its result (the next frame)
# for example, if seq_size = 10, for each frame, it will take 10 frames starting on the initial frame,
# it will create a list with those 10 sequential frames, and the 11th frame will be returned in the result list
def createSequenceFromFataset(dataset, ids, sequenceSize = 10):
    sequencedDataset = []
    sequencedDatasetResults = []
    sequencedIds = []
    for person, id in zip(dataset, ids):
        for frame in range(0, len(person)):
            end_ix = frame + sequenceSize
            if end_ix > len(person)-1:
                break
            seq_x, seq_y = person[frame:end_ix], person[end_ix]
            sequencedDataset.append(seq_x.copy())
            sequencedIds.append(id)
            sequencedDatasetResults.append(seq_y.copy())
    return sequencedDataset, sequencedDatasetResults, sequencedIds

# creates a dataset containing sequences of n frames, and the result being the next frame
def loadSequenceDataset(datasetName, partition = "All", specificSize = -1, verbose = False, sequenceSize = 10, trim = False, specificTrim = -1):
    # load the dataset in one list and the ids in the second one
    dataset, ids = loadDataset(datasetName=datasetName, partition=partition, specificSize=specificSize, verbose=verbose, trim=trim, specificTrim=specificTrim)
    # create the sequences and results
    datasetX, datasetY, sequencedIds = createSequenceFromFataset(dataset=dataset, ids=ids, sequenceSize=sequenceSize)
    return datasetX, datasetY, sequencedIds

# loads both the dataset, and its result (X, y) and returns them in two separate lists
def loadDatasetAndCreateResults(datasetName, partition = "All", specificSize = -1, verbose = False, trim = False, specificTrim = -1):
    # load the dataset in one list and the ids in the second one
    datasetX, ids = loadDataset(datasetName=datasetName, partition=partition, specificSize=specificSize, verbose=verbose, trim=trim, specificTrim=specificTrim)
    # using that frame list, create the result list
    datasetY = createResultsFromDataset(datasetX)
    # trim the last frame of the original dataset
    datasetX = trimLastFrameFromDataset(datasetX)
    # return the dataset,the results and the ids
    return datasetX, datasetY, ids

# loads only the dataset, with no results and no ids, to train the discrete VAE used as an encoder for GPT
def loadDatasetForVae(datasetName, partition = "All", specificSize = -1, verbose = False):
    datasetX = loadDatasetInBulk(datasetName=datasetName, partition=partition, specificSize=specificSize, verbose=verbose)
    return datasetX

def loadDatasetForConvolutional(datasetName, partition = "All", specificSize = -1, verbose = False):
    # load the dataset normally
    datasetX, ids = loadDataset(datasetName=datasetName, partition=partition, specificSize=specificSize, verbose=verbose)
    return datasetX, ids

if __name__ == "__main__":
    x, y, id = loadSequenceDataset("silenceDataset3sec", partition="Train", specificSize=10, trim=False, sequenceSize=30, verbose=True)
    print(np.shape(x))
    print(np.shape(y))
    print(np.shape(id))