import torch
import torch.nn as nn

N_CLASSES = 24  # Number of classes
IN_DIM = 600  # Input vectors dimensionality


# Neural network class
class FCNN(nn.Module):
    '''
    Constructor function
        hidden_dim -- number of neurons in hidden layers
        n_hidden -- number of hidden layers
        p_dropout -- dropout probability
        act -- activation function between layers
    '''
    def __init__(self, hidden_dim=100, n_hidden=1, p_dropout=0.5, act=nn.ELU):
        super().__init__()

        self.fc1 = nn.Linear(IN_DIM, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = act()
        self.do1 = nn.Dropout(p_dropout)

        # Hidden layers
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act())
            layers.append(nn.Dropout(p_dropout))

        self.hidden = nn.Sequential(*layers)

        self.fc2 = nn.Linear(hidden_dim, N_CLASSES)
        self.sm = nn.Softmax()


    '''
    Forward neural network function
        X -- input data
    Returns probability of each class for every sample 
    '''
    def forward(self, X):
        X = self.fc1(X)
        X = self.norm1(X)
        X = self.act1(X)
        X = self.do1(X)

        X = self.hidden(X)

        X = self.fc2(X)
        return self.sm(X)


'''
Function for training neural network
    net -- network
    optimizer -- parameters optimizer
    loss_fn -- loss function
    train_data -- train data loader
    n_epochs -- number of epochs to iterate over
    accuracy_checker -- function for accuracy evaluation
    verbose_int -- interval in epochs for printing training progress
    test_data -- test data loader
'''
def train_model(net, optimizer, loss_fn, train_data, n_epochs, accurcay_checker, verbose_int=5, test_data=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net.to(device)

    # Training loop
    for epoch in range(n_epochs):
        for phase in ['train', 'test']:
            if phase == 'test' and test_data == None:
                continue
            batches = 0
            acc_loss = 0
            samples = 0
            correctly_pred = 0

            loader = train_data if phase == 'train' else test_data

            if phase == 'test':
                net.eval()
            else:
                net.train()

            # Iterate over batches
            for names, labels in loader:
                batches += 1
                samples += len(labels)

                names = names.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # Reset optimizer

                # Compute loss make optimization step
                pred = net(names)

                loss = loss_fn(pred, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                acc_loss += loss.item()

                correctly_pred += accurcay_checker(labels, pred).item()
            if epoch % verbose_int == 0:
                print("Epoch", epoch, phase, "loss:", acc_loss / batches, "accuracy:", correctly_pred / samples)


'''
Run training process
    net -- network
    train_loader -- train data loader
    test_loader -- test data loader
    lr -- learning rate
    n_epochs -- number of epochs to iterate over    
'''
def run_model(net, train_loader, test_loader, lr=5e-4, epochs=100):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_model(net, optimizer, criterion, train_loader, epochs, lambda l, b: (l == (torch.argmax(b, dim=-1))).sum(), test_data=test_loader)