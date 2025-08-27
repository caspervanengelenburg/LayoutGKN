from torch_geometric.data import Dataset


def prep_data(data, device):
    """Prepares the graph data from DataBatch of the PyG Dataloader"""
    data = data.to(device)
    return (
        data.edge_index,
        data.geometry.float(),
        data.category.long(),
        data.shp.float(),
        data.connectivity.long(),
        data.batch,
    )


class TripletDataset(Dataset):
    """Triplet Graph Dataset for RPLAN."""
    def __init__(self, triplets, fold=None):
        super(TripletDataset).__init__()
        self.triplets = triplets
        self.fold = fold

    def __getitem__(self, index):
        if self.fold: return self.triplets[self.fold[index]]
        else: return self.triplets[index]

    def __len__(self):
        if self.fold: return len(self.fold)
        else: return len(self.triplets)