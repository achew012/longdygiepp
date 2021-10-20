local template = import "template.libsonnet";

#allenai/longformer-base-4096
#bert-base-uncased, 512-tokens

template.DyGIE {
  bert_model: "allenai/longformer-base-4096",
  max_wordpieces_per_sentence: 1024,
  cuda_device: 0,
  data_paths: {
    train: "data/train.jsonl",
    validation: "data/dev.jsonl",
    test: "data/test.jsonl",

    #collation screws up the format
    #train: "data/wikievents/normalized-data/train.jsonl",
    #validation: "data/wikievents/normalized-data/dev.jsonl",
    #test: "data/wikievents/normalized-data/test.jsonl",

  },
  loss_weights: {
    ner: 1.0,
    relation: 0.0,
    coref: 0.0,
    events: 0.0
  },

  target_task: "ner"
}
