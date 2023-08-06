from transformers import DetrImageProcessor, DetrForObjectDetection
from cachetools import LFUCache
from sys import maxsize

class Detect:
    def __init__(self, cache: LFUCache(maxsize)) -> None:
        self.cache = cache

    async def loadModel(self):
        self.cache["processor"], self.cache["model"] = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50"), DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

MLModel = Detect(cache=LFUCache(maxsize=10))