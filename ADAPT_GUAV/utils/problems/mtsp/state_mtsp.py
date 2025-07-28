import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter

class StateMTSP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    
    # State
    ids: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor
    
    # MTSP specific state
    num_salesmen: int
    tours: torch.Tensor # Record of tours for all salesmen
    
    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(-2))

    def __getitem__(self, key):
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
            tours=self.tours[key]
        )

    @staticmethod
    def initialize(input, num_salesmen, visited_dtype=torch.uint8):
        depot = input['depot']
        loc = input['loc']
        
        batch_size, n_loc, _ = loc.size()
        
        return StateMTSP(
            coords=torch.cat((depot, loc), -2),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=(
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 64) // 64, dtype=torch.int64, device=loc.device)
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
            num_salesmen=num_salesmen,
            tours=torch.zeros(batch_size, 0, dtype=torch.long, device=loc.device) # Initially empty tours
        )

    def get_final_cost(self):
        assert self.all_finished()
        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):
        selected = selected[:, None]
        
        # Update tour lengths
        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)
        
        # Update visited nodes
        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, selected[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, selected - 1)
            
        # When a salesman returns to the depot, we may start a new tour
        # This logic needs to be carefully designed.
        # For simplicity, we assume the model learns to partition nodes and
        # returns to depot only when a tour is complete.
        
        return self._replace(
            prev_a=selected,
            visited_=visited_,
            lengths=lengths,
            cur_coord=cur_coord,
            i=self.i + 1
        )

    def all_finished(self):
        # All nodes visited, and the last action for each salesman is to return to depot
        return self.i.item() >= (self.coords.size(-2) - 1 + self.num_salesmen)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        # Mask visited nodes
        visited_loc = self.visited[:, :, 1:] if self.visited_.dtype == torch.uint8 else mask_long2bool(self.visited_, n=self.coords.size(-2) -1)
        
        # Depot can be visited if not all nodes are visited yet
        mask_loc = visited_loc
        
        # Prevent returning to depot if just at depot, unless all nodes are visited
        mask_depot = (self.prev_a == 0) & (visited_loc.int().sum(-1) < visited_loc.size(-1))
        
        return torch.cat((mask_depot[:, :, None], mask_loc), -1).bool()

    def construct_solutions(self, actions):
        return actions