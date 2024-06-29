import torch
import time
import os


def cls_fit(model, train_dataloader, val_dataloader, epochs, loss_func, optimizer, device):
    """
    Train and validate a classification model.

    Args:
        model: The model to train.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        epochs (int): Number of epochs to train.
        loss_func: Loss function to use for training.
        optimizer: Optimizer to use for training.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
    """
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        since = time.time()
        train_loss = 0
        val_loss = 0
        
        # Training
        model.train()
        for data, labels in train_dataloader:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            for data, labels in val_dataloader:
                data, labels = data.to(device), labels.to(device)
                
                outputs = model(data)
                loss = loss_func(outputs, labels)
                
                val_loss += loss.item()
        
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Time {time.time()-since:.2f}s, Train Loss {train_loss:.5f}, Val Loss {val_loss:.5f}')
        
    return train_losses, val_losses


def cls_acc(model, val_dataloader, device):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
    
    return accuracy


def model_latency(model, device, input_size=(1, 3, 224, 224)):
    """
    Measures the average latency of a model running on a specific device.

    Args:
        model (torch.nn.Module): The model to measure.
        device (str): The device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        float: The average latency of the model.
    """
    if not torch.cuda.is_available() and device == 'cuda':
        raise ValueError("CUDA is not available but 'cuda' was specified as the device.")
    
    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(10):  # 预热模型，通过运行模型10次来预热GPU，减少首次运行时的开销对测量结果的影响
            _ = model(x)
    torch.cuda.synchronize()   # 在测量延迟之前，确保所有CUDA操作都完成，以获得更准确的时间测量
    
    # 再次使用torch.no_grad来禁用梯度计算，确保我们的延迟测量不受梯度计算的影响
    with torch.no_grad():
        start_time = time.perf_counter()  # 记录开始时间，使用高精度的perf_counter来测量时间
        for _ in range(100):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.perf_counter()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算总的运行时间
    elapsed_time_ave = elapsed_time / 100  # 计算平均延迟，即总运行时间除以迭代次数

    return elapsed_time_ave  # 返回计算出的平均延迟
