import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import scipy.io

np.random.seed(42)
torch.manual_seed(42)

npy_data = np.load('/data/hdd0/yuteng.liu/Secpath_DNN/coordinates.npy')

input_data = npy_data.astype(np.float32)
# print(input_data[:100,:])
mat_data = scipy.io.loadmat('/data/hdd0/yuteng.liu/Secpath_DNN/2025_comsol_data_ver2/pri_path_all.mat')
data = mat_data['h'].astype(np.float32)
min_value = np.min(data)
max_value = np.max(data)
target_data = (data - min_value) / (max_value - min_value)

# print(target_data.shape)
ip_train, ip_test, op_train, op_test = train_test_split(input_data, data, test_size=0.1, random_state=42)
np.save('/data/hdd0/yuteng.liu/Secpath_DNN/2025_comsol_data_ver2/primary_path/ip_test.npy', ip_test)
np.save('/data/hdd0/yuteng.liu/Secpath_DNN/2025_comsol_data_ver2/primary_path/ip_train.npy', ip_train)
np.save('/data/hdd0/yuteng.liu/Secpath_DNN/2025_comsol_data_ver2/primary_path/op_pri_test.npy', op_test)
scipy.io.savemat('/data/hdd0/yuteng.liu/Secpath_DNN/2025_comsol_data_ver2/primary_path/op_pri_test.mat', {'test_label_pri_all':op_test})

X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.1, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_test)
np.save('/data/hdd0/yuteng.liu/Secpath_DNN/2025_comsol_data_ver2/primary_path/input_test.npy', X_test)
np.save('/data/hdd0/yuteng.liu/Secpath_DNN/2025_comsol_data_ver2/primary_path/target_test.npy', y_test)
class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, y_true, y_pred):

        y_true = torch.Tensor(y_true)
        y_pred = torch.Tensor(y_pred)

        mse = torch.norm(y_true - y_pred)

        fenmu = torch.norm(y_true)

        nmse_score = mse / fenmu

        return nmse_score
    
class ThreeLayerDNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(ThreeLayerDNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.activation1 = nn.ReLU() 
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size2, hidden_size3)
        self.activation3 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        x = self.activation3(x)
        x = self.output_layer(x)
        return x


def train(model, optimizer, criterion, train_loader, val_loader, num_epochs=10):
    train_losses = []
    val_losses = []
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        train_loss = evaluate(model, criterion, train_loader)
        val_loss = evaluate(model, criterion, val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()


        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses, best_model

def evaluate(model, criterion, data_loader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(data_loader)

X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f'Fold {fold+1}/{num_folds}')

    X_train_fold, y_train_fold = X_train[train_index], y_train[train_index]
    X_val_fold, y_val_fold = X_train[val_index], y_train[val_index]

    train_loader_fold = torch.utils.data.DataLoader(list(zip(X_train_fold, y_train_fold)), batch_size=4, shuffle=True)
    val_loader_fold = torch.utils.data.DataLoader(list(zip(X_val_fold, y_val_fold)), batch_size=2, shuffle=False)

    input_size = 4
    hidden_size1 = 64
    hidden_size2 = 128
    hidden_size3 = 256
    output_size = 256

    model = ThreeLayerDNN(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = NMSELoss()

    train_losses, val_losses, best_model = train(model, optimizer, criterion, train_loader_fold, val_loader_fold, num_epochs=300)
    save_model_path = f'/data/hdd0/yuteng.liu/Secpath_DNN/2025_comsol_data_ver2/primary_path/best_model_{fold+1}_bs_4_0522_2'
    torch.save(best_model, save_model_path)

    model.load_state_dict(torch.load(save_model_path))

    test_loss = evaluate(model, criterion, torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False))
    print(f'Test Loss: {test_loss:.4f}\n')