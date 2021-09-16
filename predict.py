from clearml import Task, StorageManager, Dataset
import dygie_api as dygie
import os, sys, json

task = Task.init("Dygiepp", "predict")
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

dataset = Dataset.get(dataset_name="wikievents-dygiepp-fmt", dataset_project="datasets/wikievents", dataset_tags=["dygiepp"], only_published=True)
dataset_folder = dataset.get_local_copy()
current_dir = os.getcwd()
# os.symlink(os.path.join(dataset_folder, "data", "data"), "{}/data".format(current_dir))
os.symlink(os.path.join(dataset_folder, "data", "upload"), "{}/data".format(current_dir))
sys.path.append(current_dir)

test_set = dygie.read_json(os.path.join(current_dir,"data/test.jsonl"))
results = dygie.run_dataset(test_set, pretrained_model_path='../data/dygiepp/models/acepp_robertabase.tar.gz', spacy_model="en_core_web_md", model_type="ace-event-plus-rams")

task.upload_artifact('predictions', artifact_object=results)
task.close()

