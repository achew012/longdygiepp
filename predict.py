from clearml import Task, StorageManager, Dataset
import os, sys, json


task = Task.init("Dygiepp", "predict", output_uri="s3://experiment-logging/storage/")
task.set_base_docker("nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04")
task.execute_remotely(queue_name="128RAMv100", exit_process=True)

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

bucket_ops.download_folder(
    local_path="/models/dygiepp/", 
    remote_path="s3://experiment-logging/storage/Dygiepp/train-10events.e82dc97d1e814b4fb89fa2318b36b1c3/artifacts/model/", 
)

# bucket_ops.download_folder(
#     local_path="/models/dygiepp", 
#     remote_path="s3://experiment-logging/pretrained/ace-doc", 
# )

print(list(os.walk("/models/dygiepp")))

dataset = Dataset.get(dataset_name="wikievents-10events", dataset_project="datasets/wikievents", dataset_tags=["dygiepp"], only_published=True)
dataset_folder = dataset.get_local_copy()
current_dir = os.getcwd()
# os.symlink(os.path.join(dataset_folder, "data", "data"), "{}/data".format(current_dir))
os.symlink(os.path.join(dataset_folder, "data", "upload"), "{}/data".format(current_dir))
sys.path.append(current_dir)

import dygie_api as dygie

test_set = dygie.read_json(os.path.join(current_dir,"data/test.jsonl"))
# test_set = [{**doc, 'dataset': 'dwie'} for doc in test_set]
results = dygie.run_dataset(test_set, pretrained_model_path="/models/dygiepp/model.tar.gz", spacy_model="en_core_web_md", model_type="wikievent")
dygie.to_jsonl("predictions.jsonl", results)

task.upload_artifact('predictions', artifact_object="predictions.jsonl")
task.close()

