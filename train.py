from clearml import Task, StorageManager, Dataset
import json

def create_clearml_dataset(dataset_project, dataset_version, local_dataset_path:str, tags=[]):
    dataset = Dataset.create(dataset_name=dataset_version, dataset_project=dataset_project, dataset_tags=tags, parent_datasets=None, use_current_task=False)
    logger = dataset.get_logger() # for logging of media to debug samples
    dataset.add_files(local_dataset_path, wildcard="*.json", local_base_folder=".", dataset_path="data", recursive=True, verbose=False)
    dataset.upload(output_url="s3://experiment-logging/datasets")
    dataset.finalize()
    dataset.publish()
    print("Files added", dataset.list_files())
    return dataset

def delete_dataset(dataset_project, dataset_name):
    Dataset.delete(dataset_name=dataset_name, dataset_project=dataset_project, force=True)

#delete_dataset(dataset_name="processed-data", dataset_project="ace05-event")
#dataset = create_clearml_dataset("ace05-event", "processed-data", "data/ace-event/processed-data/default-settings/json/", tags = ["original"])
#dataset = create_clearml_dataset("ace05-event", "collated-data", "data/ace-event/collated-data/default-settings/json/", tags = ["original"])

class bucket_ops:
    StorageManager.set_cache_file_limit(5, cache_context=None)

    def list(remote_path:str):
        return StorageManager.list(remote_path, return_full_path=False)

    def upload_folder(local_path:str, remote_path:str):
        StorageManager.upload_folder(local_path, remote_path, match_wildcard=None)
        print("Uploaded {}".format(local_path))

    def download_folder(local_path:str, remote_path:str):
        StorageManager.download_folder(remote_path, local_path, match_wildcard=None, overwrite=True)
        print("Downloaded {}".format(remote_path))
    
    def get_file(remote_path:str):        
        object = StorageManager.get_local_copy(remote_path)
        return object

    def upload_file(local_path:str, remote_path:str):
        StorageManager.upload_file(local_path, remote_path, wait_for_upload=True, retries=3)

task = Task.init("Dygiepp", "longdygiepp")
task.execute_remotely(queue_name="default", exit_process=True)

# Download Pretrained Models

# bucket_ops.download_folder(
#     local_path="./pretrained/longformer-base-4096", 
#     remote_path="s3://experiment-logging/pretrained/longformer-base-4096", 
#     )

import os, subprocess, sys

dataset = Dataset.get(dataset_name="wikievents-dygiepp-fmt", dataset_project="datasets/wikievents", dataset_tags=["dygiepp"], only_published=True)
dataset_folder = dataset.get_local_copy()
# if os.path.exists(dataset_folder)==False:
# os.symlink(os.path.join(dataset_folder, "data", "data"), "{}/data".format(os.getcwd()))
os.symlink(os.path.join(dataset_folder, "data", "upload"), "{}/data".format(os.getcwd()))

current_dir = os.getcwd()
train_script = os.path.join(current_dir, "scripts/train.sh")
sys.path.append(current_dir)

# Use this to initiate the dataset/dataloader objects
#dataset_paths = [os.path.join(dataset_folder, "data/train.json"), os.path.join(dataset_folder, "data/dev.json"), os.path.join(dataset_folder, "data/test.json")]

#os.chmod(train_script, 0o755)
#subprocess.run(["ls", "data"])
subprocess.run(["pip", "install", "-r", "requirements.txt"])
subprocess.run(["bash", train_script, "wikievent"])

task.upload_artifact('model', artifact_object=os.path.join("models", "wikievent", "model.tar.gz"))
task.upload_artifact('train_metrics', artifact_object=os.path.join('models', 'wikievent', "metrics.json"))

task.close()



