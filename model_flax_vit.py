from typing import TYPE_CHECKING, Tuple

import numpy as np

import jax
from jax.experimental import maps, pjit, PartitionSpec

from transformers import ViTFeatureExtractor, FlaxViTForImageClassification
from PIL import Image
from rich.progress import Progress, TimeElapsedColumn

if TYPE_CHECKING:
    from transformers.modeling_flax_outputs import FlaxSequenceClassifierOutput

TEXT_LABEL = str


class ViTPipeline:
    """
    ViT pipeline with pjit.
    Initializing the class will initialize and activate
    the JAX mesh.
    """

    def __init__(self, hugging_face_model_name: str) -> None:
        devices = np.array(jax.devices())  # (num_devices,)

        # Defaults to a batch size of 1. (No data parallelism.)
        # Model is parallelized across all accelerator devices.
        self.device_mesh = devices.reshape((-1, 1))  # (num_devices, 1)

        # Device mesh: (model parallelism axis, data parallelism axis).
        self.mesh_context = maps.Mesh(self.device_mesh, ("mp", "dp"))

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            hugging_face_model_name
        )
        (
            self.model,
            self.model_params,
        ) = FlaxViTForImageClassification.from_pretrained(  # type: ignore
            hugging_face_model_name,
            dtype=jax.numpy.float16,  # type: ignore
            _do_init=False,
        )
        self.pjit_model_fn = pjit.pjit(
            self.model.__call__,
            # Replicate input image, model params, and output labels.
            # Exact ops placement is up to pjit.
            # Adjust in_axis_resources to support larger models.
            in_axis_resources=PartitionSpec(),
            out_axis_resources=PartitionSpec(),
        )
        self.labels = self.model.config.id2label
        if self.labels is None:
            self.labels = {}

    def predict(self, image: Image.Image) -> Tuple[int, TEXT_LABEL]:
        """
        Run pipeline on the given image and return integer label.
        """
        self.mesh_context.__enter__()
        rgb_image = image.convert("RGB")
        pixel_values = self.feature_extractor(rgb_image)["pixel_values"]
        pixel_values_array = jax.numpy.array(pixel_values)
        vit_output = self.pjit_model_fn(pixel_values_array, self.model_params)
        vit_output: FlaxSequenceClassifierOutput
        vit_prediction: int = jax.numpy.argmax(vit_output.logits).item()

        label = self.labels.get(vit_prediction, "")
        return vit_prediction, label


def load_saved_model(hugging_face_model_name) -> ViTPipeline:
    with Progress(*Progress.get_default_columns(), TimeElapsedColumn()) as progress:
        load_params_progress_bar = progress.add_task("Loading weights")
        pipeline = ViTPipeline(hugging_face_model_name)
        progress.update(load_params_progress_bar, total=1, completed=1)

        # Pre-compile pjit model to avoid delay
        # at the first prediction request.
        precompile_progress_bar = progress.add_task("Precompiling model")
        example_image_array = np.zeros((256, 256))
        example_image = Image.fromarray(example_image_array)
        _ = pipeline.predict(example_image)
        progress.update(precompile_progress_bar, total=1, completed=1)

    return pipeline
