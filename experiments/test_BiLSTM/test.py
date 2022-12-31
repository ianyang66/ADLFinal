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
TOP_K = 30

seen_unseen = 'unseen'

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
        return id_list,torch.stack(input_list)

class SubgroupPredictions(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        #dropout: float,
    ) -> None:
        super(SubgroupPredictions, self).__init__()
        self.lstm = nn.LSTM(NUM_SUBGROUP, 
                    hidden_size,
                    bidirectional=True,
                    num_layers=num_layers)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(2*hidden_size, NUM_SUBGROUP)
        #self.dropout = nn.Dropout(0.5)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        out, (h, c) = self.lstm(batch)
        out = self.linear(out)
        out = self.softmax(out)
        return out

def main(args):
    #user_id, input, label
    test_data = get_data('test', seen_unseen)
    test_dataset = SubgroupDataset(test_data)

    test_dl = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SubgroupPredictions(hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=torch.device(device))
    model.load_state_dict(ckpt)    
   
    predictions = []
    model.eval()

    with torch.no_grad():
        for id, inputs in test_dl:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            predicted = torch.argsort(outputs.data, axis=1, descending=True)

            for i, p in zip(id, predicted):
                predictions.append([i, ' '.join([str(x) for x in p[:TOP_K].tolist()])])

    with open('./possible_solution/submit_' + seen_unseen +'.csv', 'w') as f:
        f.write('user_id,subgroup\n')
        for pre in predictions:
            f.write('{},{}\n'.format(pre[0], pre[1]))
    






def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/lstm.pt",
    )

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=20)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)