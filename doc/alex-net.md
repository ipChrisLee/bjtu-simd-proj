# alex-net

## alex net structure

[This article](https://zhuanlan.zhihu.com/p/662953988) introduces the structure of alexnet. Core python code is:

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
        	nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        	nn.ReLU(inplace=True),
        	nn.MaxPool2d(kernel_size=3, stride=2),
        	nn.Conv2d(64, 192, kernel_size=5, padding=2),
        	nn.ReLU(inplace=True),
        	nn.MaxPool2d(kernel_size=3, stride=2),
        	nn.Conv2d(192, 384, kernel_size=3, padding=1),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(384, 256, kernel_size=3, padding=1),
        	nn.ReLU(inplace=True),
        	nn.Conv2d(256, 256, kernel_size=3, padding=1),
        	nn.ReLU(inplace=True),
        	nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
        	nn.Dropout(),
        	nn.Linear(256 * 6 * 6, 4096),
        	nn.ReLU(inplace=True),
        	nn.Dropout(),
        	nn.Linear(4096, 4096),
        	nn.ReLU(inplace=True),
        	nn.Linear(4096, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

[This article](https://zhuanlan.zhihu.com/p/467017218) also introduces the alexnet, but with detailed info on how every layer constucted, but with little difference. In this article, alex contains LRN layer, which is not mentioned in the previous article.

## some layer intro

* [pytorch conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).
* [pytorch maxpoll](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html).
