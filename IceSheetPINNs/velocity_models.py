from pinns.models import INR
import torch.nn as nn


class INR_Velocity(nn.Module):
    def __init__(
        self,
        name,
        input_size,
        output_size,
        hp,
    ):
        self.inr_pos = INR(name, input_size, output_size, hp)
        self.inr_velcoity = INR(name, input_size, 3, hp)

    def forward(self, *args):
        return self.inr_pos(args), self.inr_velcoity(args)
