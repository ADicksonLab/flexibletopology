import torch
from torch.autograd import Variable
from pharmostat.layers.GSGraph import GSGraph


# coords = torch.tensor([[0.7852, 0.3557, 0.9492],
#         [0.6310, 0.2493, 0.6343],
#         [0.6927, 0.1372, 0.3046],
#         [0.4866, 0.7781, 0.1464],
#         [0.2730, 0.1045, 0.8313]])


# signals = torch.tensor([[0.3447, 0.9564, 0.8883, 0.7758],
#         [0.5669, 0.1726, 0.1882, 0.9085],
#         [0.2121, 0.7749, 0.6671, 0.6151],
#         [0.9452, 0.6738, 0.7336, 0.5276],
#         [0.6060, 0.4118, 0.7541, 0.3233]])


coords = torch.rand(7, 3)*10
signals = torch.rand(7, 3)*2
model = GSGraph()
device = torch.device('cpu')
model.to(device)


coords = Variable(coords, requires_grad=True)

signals = Variable(signals, requires_grad=True)

n_in = 7
# coords = torch.randn((n_in, 3)*10, requires_grad=True, device='cpu')
# signals = torch.randn((n_in, 4)*2, requires_grad=True, device='cpu')
y_test = model(coords, signals)

print("pharmo", y_test.shape)
loss_fn = torch.nn.MSELoss()

loss = loss_fn(y_test, torch.rand_like(y_test))

loss.backward(retain_graph=True)
print(coords.grad)
print(signals.grad)



# example=torch.randn((n_in, 3), dtype=torch.double, requires_grad=True, device='cpu')


# #Save the trained model
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("torch_model.pt")

# #import ipdb; ipdb.set_trace()
# model.eval()
# y_test = model(example)
# y_test.backward()
# print(example.grad.shape)
# #d = torch.autograd.grad(y_test, example)[0]
# #print(d.shape)
