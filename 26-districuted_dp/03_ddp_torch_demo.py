
# torchrun --nproc-per-node=2 03_ddp_torch_demo.py
# use V100*2 to train

import os
		
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import torch
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification

class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)
    
dist.init_process_group(backend="nccl")

data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
data = data.dropna()


dataset = MyDataset()
trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))
print(f'trainset size : {len(trainset)}, validset size : {len(validset)}')
for i in range(5):
    print(trainset[i])

tokenizer = BertTokenizer.from_pretrained("/tmp/pretrainmodel/chinese-roberta-wwm-ext")

def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs

trainloader = DataLoader(trainset, batch_size=128, collate_fn=collate_func, sampler=DistributedSampler(trainset))
validloader = DataLoader(validset, batch_size=256, collate_fn=collate_func, sampler=DistributedSampler(validset))

model = BertForSequenceClassification.from_pretrained("/tmp/pretrainmodel/chinese-roberta-wwm-ext")

if torch.cuda.is_available():
    model = model.to(int(os.environ["LOCAL_RANK"]))

model = DDP(model)

optimizer = Adam(model.parameters(), lr=2e-5)

def print_rank_0(info):
    if int(os.environ["RANK"]) == 0:
        print(info)

def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    dist.all_reduce(acc_num)
    return acc_num / len(validset)

def train(epoch=5, log_step=20):
    global_step = 0
    for ep in range(epoch):
        model.train()
        trainloader.sampler.set_epoch(ep)
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                print_rank_0(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        acc = evaluate()
        print_rank_0(f"ep: {ep}, acc: {acc}")

train()
