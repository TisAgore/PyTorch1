---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .code execution_count="17" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="g7w8F738RG_7" outputId="4eefcabe-bdfe-446c-a8bd-1d68a7647535"}
``` python
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

::: {.output .stream .stdout}
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
:::
:::

::: {.cell .code execution_count="18" id="0pLu7vweVnA3"}
``` python
import torch as t
```
:::

::: {.cell .markdown id="F6ZlDVQPUkjK"}
## 1.1 Создание тензоров {#11-создание-тензоров}
:::

::: {.cell .code execution_count="19" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="QQR49ERgTsgS" outputId="e3a1b2ce-a2b5-415f-8153-4ded1b691f25"}
``` python
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
tensor3x4 = t.randint(2, (3, 4))
print('tensor3x4\n', tensor3x4)
# - Тензор размером 2x3x4, заполненный нулями
zeros_tensor2x3x4 = t.zeros(2, 3, 4)
print('zeros_tensor2x3x4\n', zeros_tensor2x3x4)
# - Тензор размером 5x5, заполненный единицами
ones_tensor_5x5 = t.ones(5, 5)
print('ones_tensor5x5\n', ones_tensor_5x5)
# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
tensor4x4 = t.randint(16, (2, 8)).reshape(4, 4)
print('tensor4x4\n', tensor4x4)
```

::: {.output .stream .stdout}
    tensor3x4
     tensor([[1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1]])
    zeros_tensor2x3x4
     tensor([[[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]],

            [[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]]])
    ones_tensor5x5
     tensor([[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]])
    tensor4x4
     tensor([[ 3, 13, 12,  0],
            [10,  6, 10,  2],
            [ 8,  9, 13, 12],
            [14, 14, 13,  7]])
:::
:::

::: {.cell .markdown id="0Qa2lKnuW6LF"}
## 1.2 Операции с тензорами {#12-операции-с-тензорами}
:::

::: {.cell .code execution_count="20" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="V2l0NyW1W8B8" outputId="04cfe227-189b-4dff-c1a2-73922856938e"}
``` python
A = t.randint(11, (3, 4))
B = t.randint(11, (4, 3))
print('A\n', A)
print('B\n', B)
# - Транспонирование тензора A
transp_tensor_A = A.T
print('transp_tensor_A\n', transp_tensor_A)
# - Матричное умножение A и B
mult_AB_tensor = A @ B
print('mult_AB_tensor\n', mult_AB_tensor)
# - Поэлементное умножение A и транспонированного B
mult_ABtransp_tensor = A * B.T
print('mult_AandBtrans_tensor\n', mult_ABtransp_tensor)
# - Вычислите сумму всех элементов тензора A
sum_A = A.sum()
print('sum_A\n', sum_A)
```

::: {.output .stream .stdout}
    A
     tensor([[1, 4, 0, 4],
            [8, 6, 2, 1],
            [7, 9, 6, 2]])
    B
     tensor([[10,  9,  1],
            [ 8,  2, 10],
            [10,  5,  4],
            [ 4,  6,  4]])
    transp_tensor_A
     tensor([[1, 8, 7],
            [4, 6, 9],
            [0, 2, 6],
            [4, 1, 2]])
    mult_AB_tensor
     tensor([[ 58,  41,  57],
            [152, 100,  80],
            [210, 123, 129]])
    mult_AandBtrans_tensor
     tensor([[10, 32,  0, 16],
            [72, 12, 10,  6],
            [ 7, 90, 24,  8]])
    sum_A
     tensor(50)
:::
:::

::: {.cell .markdown id="irw67nKKYhQE"}
## 1.3 Индексация и срезы {#13-индексация-и-срезы}
:::

::: {.cell .code execution_count="21" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="CRcu1Y2hYmqy" outputId="65c45033-53f9-469b-db43-53c367644dfc"}
``` python
tensor5x5x5 = t.randint(11, (5, 5, 5))
print('tensor5x5x5\n', tensor5x5x5)
# - Первую строку
first_string = tensor5x5x5[:1, :1, :]
print('first_string\n', first_string)
# - Последний столбец
last_column = tensor5x5x5[-1:, :, -1:]
print('last_column\n', last_column)
# - Подматрицу размером 2x2 из центра тензора
center_2x2 = tensor5x5x5[2, 1:3, 1:3]
print('center_2x2\n', center_2x2)
# - Все элементы с четными индексами
even_elements = tensor5x5x5[::2, ::2, ::2]
print('even_elements\n', even_elements)
```

::: {.output .stream .stdout}
    tensor5x5x5
     tensor([[[ 6,  9,  4,  8,  4],
             [ 0,  3,  9,  9,  9],
             [ 7,  6,  0,  9,  6],
             [ 2,  7,  0,  5,  9],
             [ 1,  0,  4,  9,  0]],

            [[ 2,  8, 10,  8,  5],
             [ 0,  2,  6,  3,  3],
             [ 9,  5,  7,  3,  2],
             [ 5,  6,  4,  5,  6],
             [ 2,  8,  9,  5,  0]],

            [[ 1,  7,  2,  3,  9],
             [ 7, 10,  9,  2,  3],
             [ 9,  2,  9,  7,  4],
             [ 1,  8,  6,  4,  6],
             [ 3,  9,  2,  3,  7]],

            [[ 6,  8,  1,  0,  2],
             [ 0,  9,  9,  7,  4],
             [ 7,  5,  1,  0,  1],
             [ 0,  9, 10,  3,  7],
             [ 4,  6,  1,  3,  3]],

            [[ 0,  5,  0,  4,  5],
             [ 7,  7,  9,  7,  8],
             [ 9, 10, 10,  7,  7],
             [ 1,  2,  2, 10,  8],
             [ 8,  2,  6,  4,  5]]])
    first_string
     tensor([[[6, 9, 4, 8, 4]]])
    last_column
     tensor([[[5],
             [8],
             [7],
             [8],
             [5]]])
    center_2x2
     tensor([[10,  9],
            [ 2,  9]])
    even_elements
     tensor([[[ 6,  4,  4],
             [ 7,  0,  6],
             [ 1,  4,  0]],

            [[ 1,  2,  9],
             [ 9,  9,  4],
             [ 3,  2,  7]],

            [[ 0,  0,  5],
             [ 9, 10,  7],
             [ 8,  6,  5]]])
:::
:::

::: {.cell .markdown id="6ItPuwwlZbI7"}
## 1.4 Работа с формами {#14-работа-с-формами}
:::

::: {.cell .code execution_count="22" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="bEeKcz1WZcDp" outputId="9b615f12-de4b-429f-93d8-5bf99434f63f"}
``` python
# Создайте тензор размером 24 элемента
tensor1x24 = t.randint(11, (24, ))
print('tensor1x24\n', tensor1x24)
# Преобразуйте его в формы:
# - 2x12
tensor2x12 = tensor1x24.reshape(2, 12)
print('tensor_2x12\n', tensor2x12)
# - 3x8
tensor3x8 = tensor1x24.reshape(3, 8)
print('tensor3x8\n', tensor3x8)
# - 4x6
tensor4x6 = tensor1x24.reshape(4, 6)
print('tensor4x6\n', tensor4x6)
# - 2x3x4
tensor2x3x4 = tensor1x24.reshape(2, 3, 4)
print('tensor2x3x4\n', tensor2x3x4)
# - 2x2x2x3
tensor2x2x2x3 = tensor1x24.reshape(2, 2, 2, 3)
print('tensor2x2x2x3\n', tensor2x2x2x3)
```

::: {.output .stream .stdout}
    tensor1x24
     tensor([ 7,  5,  3,  8, 10,  1,  3,  4,  8,  3, 10,  7, 10,  9, 10,  6,  3,  8,
            10,  4,  1, 10,  2, 10])
    tensor_2x12
     tensor([[ 7,  5,  3,  8, 10,  1,  3,  4,  8,  3, 10,  7],
            [10,  9, 10,  6,  3,  8, 10,  4,  1, 10,  2, 10]])
    tensor3x8
     tensor([[ 7,  5,  3,  8, 10,  1,  3,  4],
            [ 8,  3, 10,  7, 10,  9, 10,  6],
            [ 3,  8, 10,  4,  1, 10,  2, 10]])
    tensor4x6
     tensor([[ 7,  5,  3,  8, 10,  1],
            [ 3,  4,  8,  3, 10,  7],
            [10,  9, 10,  6,  3,  8],
            [10,  4,  1, 10,  2, 10]])
    tensor2x3x4
     tensor([[[ 7,  5,  3,  8],
             [10,  1,  3,  4],
             [ 8,  3, 10,  7]],

            [[10,  9, 10,  6],
             [ 3,  8, 10,  4],
             [ 1, 10,  2, 10]]])
    tensor2x2x2x3
     tensor([[[[ 7,  5,  3],
              [ 8, 10,  1]],

             [[ 3,  4,  8],
              [ 3, 10,  7]]],


            [[[10,  9, 10],
              [ 6,  3,  8]],

             [[10,  4,  1],
              [10,  2, 10]]]])
:::
:::
