import torch


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define top_k_error_rate
def top_k_error_rate(output, target, k=5):
    """Given the output and the target, we compute the top-1 and top-5 error rates."""
    with torch.no_grad():
        max_k = max(1, k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in range(1, max_k + 1):
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            accuracy_k = correct_k.mul_(100.0 / batch_size)
            error_rate_k = 100.0 - accuracy_k.item()
            res.append(error_rate_k)
        return res


# Define validation fonction
def validate(model, criterion, val_loader):
    """Evaluate the model on the validation set."""
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    total_images = 0
    
    cumulative_top1_error_rate = 0.0
    cumulative_top5_error_rate = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            total_images += labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            # Compute top-1 and top-5 error rates
            error_rates = top_k_error_rate(outputs, labels, k=5)
            top1_error_rate = error_rates[0]  # First element is the top-1 error rate
            top5_error_rate = error_rates[-1] # Last element is the top-5 error rate (k=5)

            cumulative_top1_error_rate += top1_error_rate
            cumulative_top5_error_rate += top5_error_rate
    
    avg_val_loss = val_loss / len(val_loader)
    avg_top1_error_rate = cumulative_top1_error_rate / len(val_loader)
    avg_top5_error_rate = cumulative_top5_error_rate / len(val_loader)
    
    val_acc = correct_val / total_val * 100

    print(f"Validation Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Top-1 Error Rate: {avg_top1_error_rate:.2f}%, Top-5 Error Rate: {avg_top5_error_rate:.2f}%")
    
    return avg_val_loss, val_acc, avg_top1_error_rate, avg_top5_error_rate