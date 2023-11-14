

# reads a bvh file and separates the data from the header. Returns the header if needed, else returns only the data in a list. Can also return just the header.
# each row of the list contains a number of joint rotations
def loadBvhToList(path, returnHeader = False, returnData = True):
    ### HEADER ###
    # read and save the header in a variable
    f = open(path + ".bvh", "r")
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
    line = f.readline()
    while True:
        data.append(line)
        line = f.readline()
        if not line: break

    if returnHeader and returnData:
        return header, data
    if returnData:
        return data
    return header