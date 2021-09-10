from gans.core.data.datasets import get_cityscrapes_dataset

def test_cityscrapes_ds():
    dataset = get_cityscrapes_dataset(type="TEST")
    
    # print(next(iter(dataset)))
    # print(dataset.class_to_idx)
    # ind = dataset.class_to_idx['train']
    # n = 0
    # for i in range(len(dataset)):
    #    dataset.imgs = []
    print(len(dataset))
    print(next(iter(dataset)))


test_cityscrapes_ds()
    
