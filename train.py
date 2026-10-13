import torch.nn as nn
from torchvision import models
from common import get_loaders, train_and_eval

def build_model(name, num_classes=2, freeze=True):
    constructors = {
        "resnet18":            lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "mobilenet_v3_small":  lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT),
        "shufflenet_v2_x1_0":  lambda: models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
    }
    net = constructors[name]()
    if name == "resnet18":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()
    elif name == "mobilenet_v3_small":
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
        head_params = net.classifier[3].parameters()
    elif name == "shufflenet_v2_x1_0":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()
    if freeze:
        for p in net.parameters():
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True
    return net

if __name__ == '__main__':
    # 开启数据增强
    from torchvision import transforms
    from common import IMAGENET_MEAN, IMAGENET_STD
    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    train_loader, val_loader = get_loaders("./datasets", batch_size=16, train_transform=train_transform)
    model_name = "resnet18"
    print(f"\n=====开始训练 {model_name} =====")
    net = build_model(model_name, freeze=True)
    best_acc = train_and_eval(net, train_loader, val_loader, epochs=8, lr=1e-3, device="cpu")
    print(f"\n训练完成，最佳验证准确率 {best_acc:.4f}，权重保存在 best_model.pth")
