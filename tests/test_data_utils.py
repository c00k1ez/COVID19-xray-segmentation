import pytest

from src.data_utils import XRayDataset, load_image


def test_load_image():
    image = load_image("./tests/test_data/covid_1.png", convert_to_rgb=True)
    assert tuple(image.shape) == (256, 256, 3)

    image = load_image("./tests/test_data/covid_1.png", convert_to_rgb=False)
    assert tuple(image.shape) == (256, 256)
