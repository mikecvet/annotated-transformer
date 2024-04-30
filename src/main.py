import argparse
from transformer import Transformer
from transformers import AutoTokenizer
from datasets import load_dataset
import beam
import math
import os
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

BATCH_SIZE = 64
MODEL_STATE_PATH = "transformer_state.data"

def detect_device(device):
  """
  Depending on the provided arguments and the available device backends, returns either a CPU,
  or GPU device, where the GPU device may be either a CUDA-based GPU or Apple Silicon "MPS" based GPU.
  Default device is CPU.
  """
  if device == "cpu":
    return torch.device("cpu")
  
  if not device or device == "gpu":
    if torch.backends.mps.is_available():
      return torch.device("mps")
    elif torch.backends.cuda.is_available:
      return torch.device("cuda")

  return torch.device("cpu")  # Fallback to CPU if no GPU device is available

def load(model, path, device):
  if not path:
    path = MODEL_STATE_PATH
    if not os.path.isfile(path):
      return
  else:
    try:
      open(path, 'r')
    except Exception as e:
      print(f"Error opening model data file {path}: {e}")
      exit(1)

  print(f"loading model data from {path}")
  model.load_state_dict(torch.load(path, map_location=device))

def save(model, path):
  path = path or MODEL_STATE_PATH
  torch.save(model.state_dict(), path)
  print(f"Saved model data to {path}")

def train(
    model, 
    data_loader, 
    epochs, 
    batch_size, 
    device, 
    loss_fn,
    path,
    learning_rate=1e-3):
  
  # https://arxiv.org/pdf/1706.03762 # 5.3
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)

  model.train()

  # Number of training batches to process
  n = math.ceil(len(data_loader.dataset) / batch_size)

  for _ in range(epochs):
    epoch_loss = 0.0
    for _, batch in tqdm(enumerate(data_loader), total=n):
      optimizer.zero_grad()

      input_ids = batch["input_ids"].to(device)
      labels    = batch["labels"].to(device)
      outputs   = model(src=input_ids, tgt=labels[:, :-1])
      loss      = loss_fn(outputs.transpose(1, 2), labels[:, 1:])

      epoch_loss += loss.item()

      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
      optimizer.step()

    print(f"epoch loss: {epoch_loss}")

    save(model, path)

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--device',      type=str,   required=False, help='Specify device for workload. Defaults to GPU if supported. [cpu | gpu]')
  parser.add_argument('--load',        type=str,   required=False, help='Path to previously-saved model data')
  parser.add_argument('--lr',          type=float, required=False, help='Learning rate')
  parser.add_argument('--save',        type=str,   required=False, help='Save model data to the specified path')
  parser.add_argument('--temperature', type=float, required=False, help='Temperature scale to apply to token generation')
  parser.add_argument('--torch',       action='store_true', help='Train a pytorch transformer model instead of the default custom model')
  parser.add_argument('--train',       type=int,   required=False, help='Train model over the specified number of epochs')
  
  args = parser.parse_args()
  device = detect_device(args.device)

  tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

  if tokenizer.bos_token is None:
    # Assuming we decide to use the CLS token as BOS for a model like DistilBERT
    tokenizer.bos_token = tokenizer.cls_token

  if tokenizer.eos_token is None:
    # use the SEP token as a fallback for the EOS token, which is a common choice for models like BERT or DistilBERT, which don't use a distinct EOS token by default.
    tokenizer.eos_token = tokenizer.sep_token

  def preprocess_function(examples):
    inputs = tokenizer(examples['document'], max_length=256, truncation=True, padding='max_length', return_tensors="pt")
    labels = tokenizer(examples['summary'], max_length=96, truncation=True, padding='max_length', return_tensors="pt")

    # Ensure outputs are in the correct format for PyTorch tensors
    inputs = {k: v.numpy().tolist() for k, v in inputs.items()}  # Convert to list of integers
    labels = labels['input_ids'].numpy().tolist()  # Convert to list of integers, focusing on input_ids

    # It's important to ensure that 'labels' is not a nested list if each label is a single value.
    # If labels are sequences, they can be lists of lists.
    return {**inputs, 'labels': labels}  

  training_dataset = load_dataset('gigaword', 'default', split="train[:20000]")
  training_dataset = training_dataset.train_test_split(test_size=0.1)
  tokenized_data = training_dataset.map(preprocess_function, batched=True).with_format("torch")

  batch_size = BATCH_SIZE

  train_loader = DataLoader(tokenized_data["train"], batch_size=batch_size, shuffle=True)
  #test_loader = DataLoader(tokenized_data["test"], batch_size=batch_size)

  model = Transformer(
    tokenizer.vocab_size,
    tokenizer.vocab_size,
    dim=256,
    dim_inner=1024,
    num_attn_heads=8,
    num_layers=4,
    max_seq_len=256,
    dropout_rate=0.1,
    device=device
  ).to(device)

  print(f"Model: {model}")
  
  loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0) # ignore pad tokens

  load(model, args.load, device)

  if args.train:
    train(model, train_loader, args.train, batch_size, device, loss_fn, args.save or MODEL_STATE_PATH, learning_rate=args.lr or 0.0001)
  
  t = tokenized_data["test"][0]

  for _ in range(1):
    print(f"test struct: {t}")

    L = []
    for i in range(96):
      L.append(tokenizer.convert_ids_to_tokens(t['labels'][i].item()))
    
    print(f"labels: {L}")

    seq = beam.search(
      tokenizer, 
      model, 
      t["input_ids"].to(device), 
      tokenizer.bos_token_id, 
      tokenizer.eos_token_id,
      device,
      beam_width=12,
      temperature=args.temperature or 2.0,
      decay_repeated=True
    )

    print(f"generated sequence IDs: {seq}")
    seq = tokenizer.convert_ids_to_tokens(seq)
    print(f"tokens: {seq}")
    print(f"expected tokens (labels): {L}")

    if args.train == 0:
      break

    t = tokenized_data["test"][0]

if __name__ == '__main__':
  main()