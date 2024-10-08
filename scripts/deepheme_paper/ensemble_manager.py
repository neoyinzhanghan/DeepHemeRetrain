import ray
from train import model_create, model_predict, model_predict_batch


@ray.remote(num_gpus=1)
class EnsembleManager:
    """Manages the ensemble of models.

    --model_path
    --model
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = model_create(model_path)

    def async_predict_images_batch(self, pil_images):

        preds = []
        for i in range(5):

            preds.append(model_predict_batch(self.model, pil_images))

        average_pred = sum(preds) / len(preds)

        return len(pil_images), average_pred
