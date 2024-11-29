from pydantic import BaseModel

from ingredient_extraction.llm_clients import llm_google_client
from ingredient_extraction.llm_ocr import LlmOCR


def test_llm_ocr():
    llm = LlmOCR(llm_client=llm_google_client)

    url = "https://images.openfoodfacts.org/images/products/316/893/001/0883/ingredients_fr.7.400.jpg"
    result = llm.extract(image_url=url)

    assert isinstance(result, BaseModel)

    assert "Avoine complète (32%)" in result.ingredients
