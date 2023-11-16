import os
import numpy as np
# reads a bvh file and separates the data from the header. Returns the header if needed, else returns only the data in a list. Can also return just the header.
# each row of the list contains a number of joint rotations

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
    while True:
        data.append(line.split(" ")[:-1])
        line = f.readline().replace("\n", "")
        if not line: break
    data = [[float(s) for s in sublist] for sublist in data]
    data = np.asanyarray(data)
    if returnHeader and returnData:
        return header, data
    if returnData:
        return data
    return header

# loads the specified bvh dataset, and the partition can also be specified
# returns: data format is a list of 3 dimensions [n(number of bvh) person][m (depends on the bvh) frame][498 rotation]
def loadDataset(datasetName, partition = "All"):
    allData = []
    allIds = []
    id = 0
    if partition=="All":
        path = "../../" + datasetName + "/"
    if partition=="Train":
        path = "../../" + datasetName + "/genea2023_trn"
    if partition=="Validation":
        path = "../../" + datasetName + "/genea2023_val"
    for root, dirs, files in os.walk(path):
        for filename in files:
            allData.append(loadBvhToList(os.path.join(root, filename)))
            allIds.append(id) # TODO: IDs should not be numbers. Change to one-hot encoding or other
            id+=1
    
    return allData, np.asarray(allIds)

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

# loads both the dataset, and its result (X, y) and returns them in two separate lists
def loadDatasetAndCreateResults(datasetName, partition = "All"):
    # load the dataset in one list and the ids in the second one
    datasetX, ids = loadDataset(datasetName=datasetName, partition=partition)
    # using that frame list, create the result list
    datasetY = createResultsFromDataset(datasetX)
    # trim the last frame of the original dataset
    datasetX = trimLastFrameFromDataset(datasetX)
    # return the dataset,the results and the ids
    return datasetX, datasetY, ids

if __name__ == "__main__":
    # TODO: add the person Id to both of the datasets
    x, y, id = loadDatasetAndCreateResults("silenceDataset3sec", partition="Validation")
    print(str(x[0][0][:10]) + " . " + str(y[0][0][:10]))
    print(str(x[0][1][:10]) + " . " + str(y[0][1][:10]))
    print(str(x[0][2][:10]) + " . " + str(y[0][2][:10]))
    print(str(x[0][3][:10]) + " . " + str(y[0][3][:10]))
    print(str(x[0][4][:10]) + " . " + str(y[0][4][:10]))

