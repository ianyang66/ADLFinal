import torch
from torch.utils.data import DataLoader
from torch import nn
from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
import os
from tqdm.auto import trange
from torch.utils.data import Dataset
from typing import Dict

sys.path.insert(0, os.path.realpath(os.getcwd() + '/possible_solution/'))
from getOneHot import get_data

NUM_SUBGROUP = 91
NUM_COURSES = 728

torch.manual_seed(55)
class SubgroupDataset(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self) -> int:

        return len(self.data)

    def __getitem__(self, index) -> Dict:
        return self.data[index]
    
    def collate_fn(self, samples: Dict) -> Dict:
        id_list = [s['user_id'] for s in samples]
        input_list = [torch.from_numpy(s['input']).to(torch.float) for s in samples]
        label_list = [torch.from_numpy(s['label']).to(torch.float)  for s in samples]
        return id_list,torch.stack(input_list),torch.stack(label_list)

class SubgroupPredictions(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super(SubgroupPredictions, self).__init__()
        self.lstm = nn.LSTM(NUM_SUBGROUP, 
                    hidden_size,
                    bidirectional=True,
                    dropout=dropout, 
                    num_layers=num_layers)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(2*hidden_size, NUM_SUBGROUP)
        #self.dropout = nn.Dropout(0.5)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        out, (h, c) = self.lstm(batch)
        out = self.linear(out)
        return out

def main(args):
    #user_id, input, label
    train_data, eval_data = get_data('train'), get_data('val')
    train_dataset = SubgroupDataset(train_data)
    eval_dataset = SubgroupDataset(eval_data)

    train_dl = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=train_dataset.collate_fn)
    eval_dl = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SubgroupPredictions(hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
   
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0
        correct = 0
        train_total = 0

        for id, inputs, labels in train_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = torch.argmax(outputs.data)

            #TODO
            correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        print('Epoch {}'.format(epoch))
        print("Train accuracy: {:.3f}".format(correct / train_total))
        print("Train total Loss: {:.3f}".format(epoch_loss))

        model.eval()
        eval_correct = 0
        eval_total = 0

        with torch.no_grad():
            for id, inputs, labels in eval_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                predicted = torch.argmax(outputs.data)

                eval_correct += (predicted == labels).sum().item()
                eval_total += labels.size(0)

            eval_acc = eval_correct / eval_total
            print("Eval accuracy: {:.3f}\n".format(eval_acc))

            if(eval_acc > best_acc):
                best_acc = eval_acc
                torch.save(model.state_dict(), args.ckpt_dir /'lstm.pt')
                print("Saved model at epoch {}\n".format(epoch))


    print()





def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt",
    )

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epoch", type=int, default=200)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)