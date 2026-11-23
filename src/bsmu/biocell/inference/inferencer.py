from bsmu.vision.dnn.inferencer import ImageModelConfig

class DnnModelConfig(ImageModelConfig):
    # Whether to use argmax instead of threshold binarination and maximum selection.
    output_argmax: bool = False
