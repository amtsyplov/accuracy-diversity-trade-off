from .user_item_interactions_dataset import UserItemInteractionsDataset


class InferenceDataset(UserItemInteractionsDataset):
    """
    Iterable dataset via all user-item pairs.
    """

    def __len__(self):
        return self.no_users * self.no_items

    def __iter__(self):
        for user_id in range(self.no_users):
            user_features = self.user_features[user_id]
            user_sequence = self.get_user_sequence(self.interactions, user_id)
            user_sequence_features = self.item_features[user_sequence]
            for item_id in range(self.no_items):
                yield (
                    user_id,
                    user_features,
                    user_sequence,
                    user_sequence_features,
                    item_id,
                    self.item_features[item_id],
                )

    @classmethod
    def from_dataset(cls, dataset: UserItemInteractionsDataset):
        return cls(
            dataset.no_users,
            dataset.no_items,
            dataset.user_features,
            dataset.item_features,
            interactions=dataset.interactions,
            padding=dataset.padding,
        )
