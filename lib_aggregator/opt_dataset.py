from torch.utils.data import Dataset


class OptDataset(Dataset):
    def __init__(self, posteriors, labels):
        self.posteriors = posteriors
        self.labels = labels

    def __getitem__(self, index):
        ret_posterior = {}

        for shard, post in self.posteriors.items():
            ret_posterior[shard] = post[index]

        return ret_posterior, self.labels[index]

    def __len__(self):
        return self.labels.shape[0]
