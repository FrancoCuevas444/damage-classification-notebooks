def build_one_part_dataset(parte, preloaded_images, train_ratio=0.8, random_state=42):
    dataset = opd.PreloadedOnePartDataset(
        parte,
        preloaded_images,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_state))

    classes = dataset.classes

    dataset_sizes = {
        'train': len(train_dataset),
        'test': len(test_dataset)
    }
    
    print()
    print("#TRAIN {} #TEST {}".format(dataset_sizes["train"], dataset_sizes["test"]))
    
    