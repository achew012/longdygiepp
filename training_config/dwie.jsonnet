local template = import "template.libsonnet";

#allenai/longformer-base-4096

template.DyGIE {
  bert_model: "allenai/longformer-base-4096",
  max_wordpieces_per_sentence: 512,
  cuda_device: 0,
  data_paths: {
    train: "data/dwie/dwie_dygiepp_train.jsonl",
    validation: "data/dwie/dwie_dygiepp_test.jsonl",
    test: "data/dwie/dwie_dygiepp_test.jsonl",

  },
  loss_weights: {
    ner: 1.0,
    relation: 1.0,
    coref: 0.0,
    events: 0.0
  },

  target_task: "ner"
}
