from torch.utils.data import DataLoader
from importlib import import_module


def get_dataloader(args):
    ### import module
    m = import_module('dataset.' + args.dataset.lower())

    if (args.dataset == 'Vimeo7'):
        print('Processing Vimeo7 dataset...')
        data_train = getattr(m, 'TrainSet')(args)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        data_eval  = getattr(m, 'EvalSet')(args)
        dataloader_eval  = DataLoader(data_eval, batch_size=1, shuffle=True, num_workers=1)
        data_test  = getattr(m, 'TestSet')(args)
        dataloader_test  = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=1)
        dataloader = {'train': dataloader_train, 'eval': dataloader_eval, 'test': dataloader_test}
    elif (args.dataset == 'Reds'):
        print('Processing Reds dataset...')
        data_train = getattr(m, 'TrainSet')(args)
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        data_eval  = getattr(m, 'EvalSet')(args)
        dataloader_eval = DataLoader(data_eval, batch_size=1, shuffle=False, num_workers=1)
        data_test  = getattr(m, 'TestSet')(args)
        dataloader_test = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=1)
        dataloader = {'train': dataloader_train, 'eval': dataloader_eval, 'test': dataloader_test}
    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader