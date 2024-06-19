import torch
import os
import torch.optim as optim
from utils.evaluate import validate


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Retrieve environment variables
IMAGE_PATH_VALID = os.getenv('IMAGE_PATH_VALID')
ANNOTATION_PATH = os.getenv('ANNOTATION_PATH')
IMAGE_PATH_TRAIN = os.getenv('IMAGE_PATH_TRAIN')
MAPPING_PATH = os.getenv('MAPPING_PATH')
SAVE_PATH = os.getenv('SAVE_PATH')


# Define training function

# Get the class names from the training folder.
image_class_names = os.listdir(IMAGE_PATH_TRAIN)
if '.DS_Store' in image_class_names:
    image_class_names.remove('.DS_Store')

def train(model, criterion, train_loader, num_epochs, val_loader, weights_file_name=None, initial_lr=0.01, adjust_lr_factor=10):
    """Training loop for a model with manual learning rate adjustment."""
    
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0005)
    best_val_loss = float('inf')
    lr = initial_lr
    total_step = len(train_loader)
    
    # Load optimizer state if a weights file is provided
    if weights_file_name is not None:    
        checkpoint = torch.load(os.path.join(SAVE_PATH, weights_file_name), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        
        for param_group in optimizer.param_groups:
                lr = param_group['lr']
                
    else:
        last_epoch = 0
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Zero the parameter gradients
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if batch_idx % 20000 == 0:
                print(batch_idx, end='\t')
        
        avg_train_loss = running_loss / total_step
        train_acc = correct_train / total_train * 100
        
        val_loss, val_acc, top1_err, top5_err = validate(model, criterion, val_loader)
        
        # we adjusted manually the learning rate
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#         else:
#             lr /= adjust_lr_factor

        for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}, Top-1 Err: {:.4f}, Top-5 Err: {:.4f}'
              .format(last_epoch + epoch + 1, num_epochs, avg_train_loss, train_acc, val_loss, val_acc, top1_err, top5_err))
        
        # Save model after at least two epochs
        if epoch >= 2:
            save_path = os.path.join(SAVE_PATH, f"alexnet_epoch_{last_epoch + epoch + 1}.pt")
            torch.save({
                        'epoch': last_epoch + epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, save_path)
            
            print(f"Model saved at: {save_path}")
