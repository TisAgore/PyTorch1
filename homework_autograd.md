```python
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

    Looking in indexes: https://download.pytorch.org/whl/cu118
    Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (2.6.0+cu118)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.11/dist-packages (0.21.0+cu124)
    Requirement already satisfied: torchaudio in /usr/local/lib/python3.11/dist-packages (2.6.0+cu124)
    Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch) (3.18.0)
    Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.11/dist-packages (from torch) (4.14.0)
    Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch) (3.5)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch) (3.1.6)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch) (2025.3.2)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu11==11.8.89 in /usr/local/lib/python3.11/dist-packages (from torch) (11.8.89)
    Requirement already satisfied: nvidia-cuda-runtime-cu11==11.8.89 in /usr/local/lib/python3.11/dist-packages (from torch) (11.8.89)
    Requirement already satisfied: nvidia-cuda-cupti-cu11==11.8.87 in /usr/local/lib/python3.11/dist-packages (from torch) (11.8.87)
    Requirement already satisfied: nvidia-cudnn-cu11==9.1.0.70 in /usr/local/lib/python3.11/dist-packages (from torch) (9.1.0.70)
    Requirement already satisfied: nvidia-cublas-cu11==11.11.3.6 in /usr/local/lib/python3.11/dist-packages (from torch) (11.11.3.6)
    Requirement already satisfied: nvidia-cufft-cu11==10.9.0.58 in /usr/local/lib/python3.11/dist-packages (from torch) (10.9.0.58)
    Requirement already satisfied: nvidia-curand-cu11==10.3.0.86 in /usr/local/lib/python3.11/dist-packages (from torch) (10.3.0.86)
    Requirement already satisfied: nvidia-cusolver-cu11==11.4.1.48 in /usr/local/lib/python3.11/dist-packages (from torch) (11.4.1.48)
    Requirement already satisfied: nvidia-cusparse-cu11==11.7.5.86 in /usr/local/lib/python3.11/dist-packages (from torch) (11.7.5.86)
    Requirement already satisfied: nvidia-nccl-cu11==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch) (2.21.5)
    Requirement already satisfied: nvidia-nvtx-cu11==11.8.86 in /usr/local/lib/python3.11/dist-packages (from torch) (11.8.86)
    Requirement already satisfied: triton==3.2.0 in /usr/local/lib/python3.11/dist-packages (from torch) (3.2.0)
    Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch) (1.3.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from torchvision) (2.0.2)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.11/dist-packages (from torchvision) (11.2.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch) (3.0.2)
    


```python
import torch as t
import math
```

## 2.1 Простые вычисления с градиентами


```python
x = t.tensor(1.0, requires_grad=True)
y = t.tensor(2.0, requires_grad=True)
z = t.tensor(3.0, requires_grad=True)
# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = (x ** 2) + (y ** 2) + (z ** 2) + (2 * x * y * z)
# Найдите градиенты по всем переменным
f.backward()
print(f"Градиент по x: {x.grad}")
print(f"Градиент по y: {y.grad}")
print(f"Градиент по z: {z.grad}")
```

    Градиент по x: 14.0
    Градиент по y: 10.0
    Градиент по z: 10.0
    

## 2.2 Градиент функции потерь


```python
# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2 где y_pred = w * x + b (линейная функция)
# Найдите градиенты по w и b
def mse_with_gradients(x, y_true, w, b):
    """
    Вычисляет MSE и градиенты по параметрам w и b

    Аргументы:
    x (torch.Tensor): Входные значения
    y_true (torch.Tensor): Истинные значения
    w (torch.Tensor): Весовой коэффициент (требует градиент)
    b (torch.Tensor): Смещение (требует градиент)

    Возвращает:
    tuple: (MSE, grad_w, grad_b)
    """
    y_pred = w * x + b
    mse = t.mean((y_pred - y_true)**2)

    mse.backward()
    grad_w = w.grad
    grad_b = b.grad

    return mse.item(), grad_w, grad_b

x = t.tensor([1.0, 2.0, 3.0])
y_true = t.tensor([2.0, 4.0, 6.0])
w = t.tensor(0.5, requires_grad=True)
b = t.tensor(0.1, requires_grad=True)

mse, grad_w, grad_b = mse_with_gradients(x, y_true, w, b)
print(f"MSE: {mse}")
print(f"Градиент по w: {grad_w}")
print(f"Градиент по b: {grad_b}")
```

    MSE: 9.910000801086426
    Градиент по w: -13.600000381469727
    Градиент по b: -5.800000190734863
    

## 2.3 Цепное правило


```python
# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
# Найдите градиент df/dx
x = 2.0
f = math.sin(x**2 + 1)
df_dx = math.cos(x**2 + 1) * 2 * x

print(f"Функция = {f:.4f}")     # -0.9589
print(f"Градиент = {df_dx:.4f}")  # 1.1346

# Проверьте результат с помощью torch.autograd.grad
x = t.tensor(2.0, requires_grad=True)
f = t.sin(x**2 + 1)
f.backward()
gradient = x.grad

print(f"PyTorch Функция = {f.item():.4f}")    # -0.9589
print(f"PyTorch Градиент = {gradient:.4f}")   # 1.1346
```

    Функция = -0.9589
    Градиент = 1.1346
    PyTorch Функция = -0.9589
    PyTorch Градиент = 1.1346
    
