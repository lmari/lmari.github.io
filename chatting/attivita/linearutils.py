import matplotlib.pyplot as plt
import torch

def get_params(model, rounding=None):
    return [round(p.item(), rounding) if rounding else p.item() for p in reversed(list(model.parameters()))]


def plot(x, y, title=None, model=None):
    plt.close('all')
    xx, yy = x.numpy(), y.numpy()
    plt.plot(xx, yy, 'o')
    if title:
        plt.title(title)
    if model:
        [k0, k1] = get_params(model)
        plt.plot(xx, k0 + k1 * xx)
    plt.show()


def train(model, criterion, optimizer, x, y, num_epochs=100):
    print("epoca\tparametri\tMSE")
    for epoch in range(num_epochs):
        y1 = model(x)
        loss = criterion(y1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mse = torch.mean((y1 - y)**2)
        print(f"{epoch+1}\t{get_params(model, 2)}\t{round(mse.item(), 2)}")