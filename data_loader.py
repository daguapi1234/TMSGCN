
from torch.utils.data import Dataset, DataLoader


class eegDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


def load_dataloader(train_data, test_data, train_label, test_label, parse_args):
    train_iter = DataLoader(dataset=eegDataset(train_data, train_label),
                            batch_size= parse_args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            # pin_memory=True,
                            num_workers=parse_args.num_workers)

    test_iter = DataLoader(dataset=eegDataset(test_data, test_label),
                           batch_size=parse_args.batch_size,
                           shuffle=False,
                           # pin_memory=True,
                           num_workers=parse_args.num_workers)

    return train_iter, test_iter
