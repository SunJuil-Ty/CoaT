if __name__ == '__main__':
    import numpy as np
    import mindspore as ms
    from mindspore import Tensor
    from coat import coat_tiny
    #mindspore.set_context(mode=ms.PYNATIVE_MODE)

    model = coat_tiny(num_classes=1000)
    print(model)
    dummy_input = Tensor(np.random.rand(1, 3, 224, 224), dtype=ms.float32)
    #print(dummy_input)
    y = model(dummy_input)
    #print(y)