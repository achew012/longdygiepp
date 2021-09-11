local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "allenai/longformer-base-4096",
  max_wordpieces_per_sentence: 512,
  cuda_device: 0,
  data_paths: {
    train: "data/ace-event/document/doc_train.jsonl",
    validation: "data/ace-event/document/doc_dev.jsonl",
    test: "data/ace-event/document/doc_test.jsonl",
  },
  loss_weights: {
    ner: 1.0,
    relation: 1.0,
    coref: 0.0,
    events: 1.0
  },
  target_task: "ner",
  
// To configure batch size -  this raises an exception where minibatching is not supported for multi-document inputs
//  data_loader +: {
//    batch_size: 5
//  },
//  trainer +: {
//    optimizer +: {
//      lr: 5e-4
//    }
//  }

}
