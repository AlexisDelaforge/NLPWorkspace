from torch.utils import data

# My Code


class AllSentencesDataset(data.Dataset): # A retravailler
    def __init__(self, path, embedder):
        self.path = path
        with open(self.path) as f:
            self.len =  sum(1 for line in f)
        self.embedder = embedder

    def __getitem__(self, index):
        all_index = list(range(self.len))
        all_index.pop(index)
        all_index.pop(0)
        line = pd.read_csv(self.path, header=0, skiprows=all_index)
        line = ['<sos>']+list(tokenizer(line['Source'][0]))+['<eos>']
        return self.embedder.tensorize(line)

    def __len__(self):
        return self.len

    def name(self):
        return 'AllSentencesDataset'