import torch

def collate(data):
    """
        Collates data for pytorch dataloader
        Args:
            data  : (list)
                0 : Input
                1 : Output
    """
    batch_size = len(data)

    x = torch.tensor([data[i][0] for i in range(batch_size)])
    y = torch.tensor([data[i][1] for i in range(batch_size)])

    return x, y