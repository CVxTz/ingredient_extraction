import base64

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from ingredient_extraction.data_model import ParsingOutput


class LlmOCR:
    def __init__(
        self,
        llm_client: BaseChatModel,
        system_prompt_template: PromptTemplate = PromptTemplate(
            input_variables=["schema"],
            template="Extract the ingredients from this product picture. Ignore the new lines. Keep words as is.\n "
            "Use the following schema: {schema}",
        ),
        output_class: type(BaseModel) = ParsingOutput,
    ):
        assert set(
            system_prompt_template.input_variables
        ).issubset(
            {"schema"}
        ), "System prompt template should be included in the following input variables: categories, schema"
        self.output_class = output_class

        self.system_prompt_template = system_prompt_template
        self.system_prompt = system_prompt_template.format(
            schema=self.output_class.model_json_schema()
        )

        self.llm_ocr = llm_client.with_structured_output(self.output_class)

    def extract(self, image_url: str) -> type(BaseModel):
        assert image_url, "image url should be non-empty"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
            {"type": "text", "text": "Please use this image"},
        ]

        prediction = self.llm_ocr.invoke(
            [SystemMessage(content=self.system_prompt), HumanMessage(content=content)]
        )

        return prediction


if __name__ == "__main__":
    from ingredient_extraction.llm_clients import llm_google_client

    llm = LlmOCR(llm_client=llm_google_client)

    url = "https://images.openfoodfacts.org/images/products/316/893/001/0883/ingredients_fr.7.400.jpg"
    print(llm.extract(image_url=url))
