
from torch_snippets import inspect, show, np, torch, nn
from torchvision.models import resnet50
import requests
from PIL import Image
from torchvision import transforms as T
from torch.nn import functional as F

model = resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
model = model.eval()
url = 'https://lionsvalley.co.za/wp-content/uploads/2015/11/african-elephant-square.jpg'
original_image = Image.open(requests.get(url, stream=True)
                            .raw).convert('RGB')

original_image = np.array(original_image)
original_image = torch.Tensor(original_image)

# 1. import imagenet and assign IDs to each class
image_net_classes = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
image_net_classes = requests.get(image_net_classes).text
image_net_ids = eval(image_net_classes)
image_net_classes = {i:j for j,i in image_net_ids.items()}
    
# 2. normalize and dnormalize the image
normalize = T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                        )
denormalize = T.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],
                          [1/0.229, 1/0.224, 1/0.225]
                          )
    
def image2tensor(input):
    x = normalize(input.clone().permute(2,0,1)*255.)[None]
    return x

def tensor2image(input):
    x = (denormalize(input[0].clone()).permute(1,2,0)*255.).type(torch.uint8)
    return x    

# define func to predict on a given image
def predict_on_image(input):
    model.eval()
    show(input)
    input = image2tensor(input)
    pred = model(input)
    pred = F.softmax(pred, dim=-1)[0]
    prob, clss = torch.max(pred, 0)
    clss = image_net_ids[clss.item()]
    print(f'PREDICTION: {clss} @ {prob.item()}')
    
#define dattack func
from tqdm import trange
losses = []
def attack(image, model, target, epsilon=1e-6):
    input = image2tensor(image)
    input.requires_grad = True
    pred = model(input)
    loss = nn.CrossEntropyLoss()(pred, target)
    loss.backward()
    losses.append(loss.mean().item())
    output = input - epsilon * input.grad.sign()
    output = tensor2image(output)
    del input
    return output.detach()

# modeify image to belog to different class
modified_images = []
desired_targets = ['lemon', 'comic book', 'sax, saxophone']

# loop and specify target class in each iteration
for target in desired_targets:
    target = torch.tensor([image_net_classes[target]])
    
    #modify images to attack over increasing epoch
    image_to_attack = original_image.clone()
    for _ in trange(10):
        image_to_attack = attack(image_to_attack, model, target)
    modified_images.append(image_to_attack)
    
# modified image and corresponding classes
for image in [original_image, *modified_images]:
    predict_on_image(image)
    inspect(image)
    


    
    
        