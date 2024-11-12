import torch


class MaxState(torch.nn.Module):
    def __init__(self, hidden_dim, heads):
        super(MaxState, self).__init__()

        assert hidden_dim % heads == 0, "Hidden size must be divisible by the number of heads."

        self.head_size = hidden_dim // heads
        self.head0 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head1 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head2 = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.head_num = heads

        self.hidden = hidden_dim

    def forward(self, input_data, state=None):
        b, s, k, h = input_data.shape[0], input_data.shape[1], self.head_num, self.head_size

        out = self.head0(input_data)

        out1 = self.head1(input_data)

        out2 = self.head2(input_data)

        out = out.reshape([b, s, k, h]).permute([0, 2, 1, 3])
        out1 = out1.reshape([b, s, k, h]).permute([0, 2, 1, 3])

        out = torch.cummax((out + out1) / h ** 0.5, 2)[0]

        out = out.permute([0, 2, 1, 3])
        out1 = out1.permute([0, 2, 1, 3])
        # out2 = out2.permute([0, 2, 1, 3])
        out = out.reshape([b, s, -1])
        out1 = out1.reshape([b, s, -1])
        # out2 = out2.reshape([b, s, -1])
        # out = self.layer_nor(out)

        # out = (out + out2) * out+out1

        # out3=torch.cummax(out,1)[0]
        out = (out + out2) * out + out1

        # out = self.alpha * out * (out + out2) + (1 - self.alpha) * out1

        return out, state


class FeedForward(torch.nn.Module):
    def __init__(self, hidden_size):
        super(FeedForward, self).__init__()

        self.ffn1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.ffn2 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.gate = torch.nn.Linear(hidden_size, hidden_size * 2)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.ffn1(x)
        x2 = self.relu(self.gate(x))
        xx = x1 * x2
        x = self.ffn2(xx)
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(DecoderLayer, self).__init__()

        self.self_attention = MaxState(hidden_size, num_heads)

        self.ffn = FeedForward(hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)

        self.alpha = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, state=None, ):
        x1, state = self.self_attention(x, state)
        x = self.layer_norm(self.alpha * self.ffn(x1) + (1 - self.alpha) * x)

        return x, state


class SamOut(torch.nn.Module):
    def __init__(self, voc_size, hidden_size, num_heads, num_layers):
        super(SamOut, self).__init__()
        self.em = torch.nn.Embedding(voc_size, hidden_size, padding_idx=3)
        self.pos = torch.nn.Embedding(1024, hidden_size)

        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(hidden_size, num_heads) for _ in range(num_layers)])
        self.head = torch.nn.Linear(hidden_size, voc_size, False)

        self.down = torch.nn.ModuleList(
            [torch.nn.Linear(2 * hidden_size, hidden_size, False) for _ in range(num_layers)])

    def state_forward(self, state, pos, x):
        if state is None:
            state = [None] * len(self.decoder_layers)
        i = 0
        for ii, decoder_layer in enumerate(self.decoder_layers):
            x = self.down[i](torch.concat([torch.zeros([x.shape[0], 1, 1]).to(device) + pos, x], -1))

            x1, state[i] = decoder_layer(x, state[i])
            x = x1 + x
            i += 1
        return x, state

    def pos_forward(self, x):
        if x.shape[1] >= 1024:
            pos = self.pos(torch.arange(0, x.shape[1]).long().to(device) // 1024).unsqueeze(0)
            pos = self.pos(torch.arange(0, x.shape[1]).long().to(device) % 1024).unsqueeze(0) + pos

        else:
            pos = self.pos(torch.arange(0, x.shape[1]).long().to(device)).unsqueeze(0)
        return pos

    def forward(self, x, state=None):
        x = self.em(x)
        pos = self.pos_forward(x)
        x, state = self.state_forward(state, pos, x)

        return self.head(x), state


device = "cpu"
if __name__ == '__main__':
    net = SamOut(235, 256, 16, 4)
    net.to(device)
    net(torch.randint(0, 200, [2, 8 * 13]).to(device))
    #
