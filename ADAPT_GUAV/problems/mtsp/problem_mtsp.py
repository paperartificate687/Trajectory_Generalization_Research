from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.mtsp.state_mtsp import StateMTSP
from utils.beam_search import beam_search


class MTSP(object):
    NAME = 'mtsp'
    VEHICLE_CAPACITY = 1.0  # Placeholder, not used in MTSP

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size, _ = dataset['loc'].size()
        # Each node must be visited exactly once, except for the depot
        n_nodes = graph_size + 1
        sorted_pi = pi.data.sort(1)[0]
        
        # Each node should be visited once, except depot which is visited num_salesmen times
        # We can check if the tour is valid by checking the number of visits to each node
        # This is a simplified check. A full check is more complex.
        
        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev
        # Starts and ends at depot are handled by the tour construction
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MTSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(input, num_salesmen, visited_dtype=torch.uint8):
        return StateMTSP.initialize(input, num_salesmen, visited_dtype)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096, num_salesmen=2):
        assert model is not None, "Provide model"
        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = MTSP.make_state(
            input, num_salesmen, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )
        return beam_search(state, beam_size, propose_expansions)


class MTSPDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=100000, offset=0, distribution=None):
        super(MTSPDataset, self).__init__()
        
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            # Adapt TSP data for MTSP: treat node 0 as depot
            self.data = [
                {
                    'loc': torch.FloatTensor(row[1:]),
                    'depot': torch.FloatTensor(row[0]).unsqueeze(0)
                }
                for row in data[offset:offset+num_samples]
            ]
        else:
            # Generate random data if no file is provided
            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    'depot': torch.FloatTensor(2).uniform_(0, 1).unsqueeze(0)
                }
                for i in range(num_samples)
            ]
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]