from clearml import Task, StorageManager, Dataset
import json, ipdb, jsonlines

# def create_clearml_dataset(dataset_project, dataset_version, local_dataset_path:str, tags=[]):
#     dataset = Dataset.create(dataset_name=dataset_version, dataset_project=dataset_project, dataset_tags=tags, parent_datasets=None, use_current_task=False)
#     logger = dataset.get_logger() # for logging of media to debug samples
#     dataset.add_files(local_dataset_path, wildcard="*.json", local_base_folder=".", dataset_path="data", recursive=True, verbose=False)
#     dataset.upload(output_url="s3://experiment-logging/datasets")
#     dataset.finalize()
#     dataset.publish()
#     print("Files added", dataset.list_files())
#     return dataset

# def delete_dataset(dataset_project, dataset_name):
#     Dataset.delete(dataset_name=dataset_name, dataset_project=dataset_project, force=True)

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

Task.add_requirements("torch")
task = Task.init("Dygiepp", "muc4-train", output_uri="s3://experiment-logging/storage/")
task.execute_remotely(queue_name="128RAMv100", exit_process=True)

# Download Pretrained Models
# bucket_ops.download_folder(
#     local_path="./pretrained/longformer-base-4096", 
#     remote_path="s3://experiment-logging/pretrained/longformer-base-4096", 
#     )

import os, subprocess, sys

# dataset = Dataset.get(dataset_name="wikievents-10events", dataset_project="datasets/wikievents", dataset_tags=["10events"], only_published=True)
# dataset_folder = dataset.get_local_copy()

dataset = Dataset.get(dataset_name="muc4-processed", dataset_project="datasets/muc4", dataset_tags=["processed", "GRIT"], only_published=True)
dataset_folder = dataset.get_local_copy()

# if os.path.exists(dataset_folder)==False:
# os.symlink(os.path.join(dataset_folder, "data", "data"), "{}/data".format(os.getcwd()))
#os.symlink(os.path.join(dataset_folder, "data", "upload"), "{}/data".format(os.getcwd()))
os.remove("{}/data".format(os.getcwd()))
os.symlink(os.path.join(dataset_folder, "data/muc4-grit/processed"), "{}/data".format(os.getcwd()))

######################################################################################################################33

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096', use_fast=True)

def read_json(jsonfile):
    with open(jsonfile, 'rb') as file:
        file_object = [json.loads(sample) for sample in file]
    return file_object

''' Writes to a jsonl file'''
def to_jsonl(filename:str, file_obj):
    resultfile = open(filename, 'wb')
    writer = jsonlines.Writer(resultfile)
    writer.write_all(file_obj)


def convert_muc42dygiepp(dataset, tokenizer):
    new_dataset=[]
    for doc in dataset:
        docid = doc["docid"]
        context = doc["doctext"]

        ### Only take the 1st label of each role
        ans = [[key, doc["extracts"][key][0][0][1] if len(doc["extracts"][key])>0 else 0, doc["extracts"][key][0][0][1]+len(doc["extracts"][key][0][0][0]) if len(doc["extracts"][key])>0 else 0, doc["extracts"][key][0][0][0] if len(doc["extracts"][key])>0 else ""] for key in doc["extracts"].keys()]    
        #context_encodings = tokenizer(context, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_offsets_mapping=True, return_tensors="pt")
        context_encodings = tokenizer(context, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_offsets_mapping=True, return_tensors="pt")


        entities = []
        for label, ans_char_start, ans_char_end, mention in ans:
            sequence_ids = context_encodings.sequence_ids()

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            pad_start_idx = sequence_ids[sequence_ids.index(0):].index(None)
            offsets_wo_pad = context_encodings["offset_mapping"][0][sequence_ids.index(0):pad_start_idx]

            if ans_char_end>offsets_wo_pad[-1][1] or ans_char_start>offsets_wo_pad[-1][1]:
                ans_char_start = 0
                ans_char_end = 0

            if ans_char_start==0 and ans_char_end==0:
                token_span=[0,0]
            else:
                token_span=[]
                for idx, span in enumerate(offsets_wo_pad):
                    if ans_char_start>=span[0] and ans_char_start<=span[1] and len(token_span)==0:
                        token_span.append(idx) 

                    if ans_char_end>=span[0] and ans_char_end<=span[1] and len(token_span)==1:
                        token_span.append(idx)
                        break                        
            
            # If token span is incomplete
            if len(token_span)<2:
                ipdb.set_trace()

            if token_span!=[0,0]:
                entities.append(token_span+[label, mention])

        new_dataset.append(
            {
              "doc_key": docid,
              "dataset": "muc4",
              "sentences": [tokenizer.tokenize(context)],
              "ner": [entities]
            }
        )
    return new_dataset

def process_datasets(dataset_folder, tokenizer):
    datasets = ["train", "dev", "test"]
    for dataset in datasets:
        file = read_json(os.path.join(dataset_folder, "data/muc4-grit/processed/{}.json".format(dataset)))
        data = convert_muc42dygiepp(file, tokenizer)
        to_jsonl("./data/{}.jsonl".format(dataset), data)

process_datasets(dataset_folder, tokenizer)

###############################################################################################

current_dir = os.getcwd()
train_script = os.path.join(current_dir, "scripts/train.sh")
predict_script = os.path.join(current_dir, "scripts/predict.sh")

sys.path.append(current_dir)
# Use this to initiate the dataset/dataloader objects
#dataset_paths = [os.path.join(dataset_folder, "data/train.json"), os.path.join(dataset_folder, "data/dev.json"), os.path.join(dataset_folder, "data/test.json")]

#os.chmod(train_script, 0o755)
#subprocess.run(["ls", "data"])
subprocess.run(["pip", "install", "-r", "requirements.txt"])

#subprocess.run(["bash", train_script, "wikievent"])
subprocess.run(["bash", predict_script, "muc4"])

task.upload_artifact('model', artifact_object=os.path.join("models", "muc4", "model.tar.gz"))
task.upload_artifact('train_metrics', artifact_object=os.path.join('models', 'muc4', "metrics.json"))
task.close()

# task.upload_artifact('model', artifact_object=os.path.join("models", "wikievent", "model.tar.gz"))
# task.upload_artifact('train_metrics', artifact_object=os.path.join('models', 'wikievent', "metrics.json"))
# task.close()



