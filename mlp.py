import torch
import torch.nn as nn
import torch.nn.functional as F

with open('./names.txt', 'r') as f:
    text = f.read().splitlines()


# Hyper-parameters
embed_size = 8
hidden_size = 128

vocab = sorted(list(set(''.join(text))))
stoi = { ch:i for i,ch in enumerate(vocab) }
stoi['.'] = 0
itos = { i:ch for i,ch in enumerate(vocab) }
vocab_size = len(vocab)
block_size = 4 
output_size = vocab_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.fc1 = nn.Linear(embed_size * block_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, targets=None):
        x = self.embed(x)
        # flatten the input
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if targets is not None:
            loss = F.cross_entropy(x, targets)
            return x, loss

        return x, None

def build_dataset(words):
    X, Y = [], []
    for w in words:
        # print(w)
        context = [0] * block_size

        for c in w + '.':
            ix = stoi[c]
            X.append(context)
            Y.append(ix)

            # print(''.join(itos[i] for i in context), '---->', itos[ix])
            context = context[1:] + [ix]

    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    return X, Y

def get_batch(X, Y, batch_size=128):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    return Xtr[ix], Ytr[ix]

n1 = int(0.8 * len(text))
Xtr, Ytr = build_dataset(text[:n1])

model = MLP().to(device)

# for name, param in model.named_parameters():
#     print(name, param.shape)

print(model.embed.weight)

# write embed weights to file
with open('./weights.bin', 'wb') as f:
    f.write(model.embed.weight.cpu().data.numpy().tobytes())

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# for epoch in range(10000):
#     model.train()

#     x, y = get_batch(Xtr, Ytr)
#     _, loss = model(x, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if epoch % 100 == 0:
#         print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))


# def sample():
#     model.eval()
#     with torch.no_grad():
#         context = [0] * block_size
#         result = []
#         for _ in range(100):
#             x = torch.tensor(context).unsqueeze(0).to(device)
#             y, _ = model(x)
#             p = F.softmax(y, dim=1).squeeze(0)
#             ix = torch.multinomial(p, 1).item()
#             context = context[1:] + [ix]
#             result.append(ix)
#             if ix == 0:
#                 break
#         return ''.join(itos[i] for i in result)

# for i in range(10):
#     print(sample())
