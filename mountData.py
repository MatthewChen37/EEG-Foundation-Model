import os
from azureml.core import Workspace, Dataset, Datastore

subscription_id = '813ac1cb-7b8e-45ab-9b3c-8510079c761a'
resource_group = 'miblab_official'
workspace_name = 'miblab-official-ml'


if __name__ == "__main__":
    workspace = Workspace(subscription_id, resource_group, workspace_name)

    datastore = Datastore.get(workspace, "eegfoundationmodeldata")
    dataset = Dataset.File.from_files(path=(datastore, 'HBN-Processed'))
    mounted_path = dataset.mount("/home/hice1/mchen439/data")

    mounted_path.start()

    directories = os.listdir("/home/hice1/mchen439/data")[0:10]
    print(directories)