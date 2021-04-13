import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class EfficientDataset(CustomDataset):
    def __init__(self, encodings, labels, texts, mapper):
        super().__init__(encodings, labels)
        self.texts = texts
        self.mapper = mapper

    def __getitem__(self, idx):
        new_idx = self.mapper[self.texts[idx]]
        item = {key: torch.tensor(val[new_idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[new_idx])
        return item

    def __len__(self):
        return len(self.texts)

    @staticmethod
    def create_short_lists(list_of_text_label_pair):
        """
        Optimizes the list of texts rather than creating duplicate encodings and labels
        :param list_of_text_label_pair: list zipped from original texts and labels
        :return: new texts, new labels, and mapper of original texts
        """
        texts = []
        labels = []
        text_counter = 0
        text_to_index = {}
        for (text, label) in list_of_text_label_pair:
            if text in text_to_index:
                continue
            else:
                text_to_index[text] = text_counter
                text_counter += 1
                texts.append(text)
                labels.append(label)
        return texts, labels, text_to_index
