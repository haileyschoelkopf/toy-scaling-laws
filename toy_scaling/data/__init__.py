


def get_dataset(
    config,
    np_rng=None,
):

    task = config.data.task

    dataset = None
    # TODO: best practices for using np rng
    train_data, test_data = train_test_split(dataset, np_rng)

    train_dataloader = torch.DataLoader(train_data, config.batch_size, shuffle=True)
    test_dataloader = torch.DataLoader(test_data, config.batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def train_test_split(dataset, train_pct, np_rng):

    # TODO: implement random split of data
    return None
