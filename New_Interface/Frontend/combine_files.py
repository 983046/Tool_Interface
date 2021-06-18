def read_Combine_Data(listOfFiles, commonField='SEQN'):
    '''
       read a list of files and combine them. This accept XPT and CSV
       @listOfFiles : array contains all files
       @commonField : a common filed in all given files
       @retun : combined dataset
    '''
    datasets = []
    for file in listOfFiles:
        if file.endswith('.XPT'):
            datasets.append(pd.read_sas(file))
        else:
            datasets.append(pd.read_csv(file))

    for i, dataset in enumerate(datasets):
        if i == 0:
            merged_dataset = dataset
        else:
            merged_dataset = pd.merge(merged_dataset,
                                      dataset, on=commonField)
    return merged_dataset