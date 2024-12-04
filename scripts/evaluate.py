import json
from statistics import mean

import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from jiwer import cer, wer
from tenacity import retry, stop_after_attempt, wait_fixed

from ingredient_extraction.llm_clients import llm_google_client
from ingredient_extraction.llm_ocr import LlmOCR
from ingredient_extraction.logger import logger


@retry(stop=stop_after_attempt(20), wait=wait_fixed(30))
def get_product_image_url(product_id):
    url = f"https://world.openfoodfacts.org/api/v2/product/{product_id}.json"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()

        if (
            data["status"] == 1
            and "product" in data
            and "image_ingredients_url" in data["product"]
        ):
            image_url = data["product"]["image_ingredients_url"]
            return image_url
        else:
            logger.warning(
                f"Error: Product {product_id} not found or image URL missing."
            )
            return (
                None  # or raise an exception depending on how you want to handle this
            )

    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching data from Open Food Facts API: {e}")
        raise e
    except json.JSONDecodeError as e:
        logger.warning(f"Error decoding JSON response: {e}")
        raise e


@retry(stop=stop_after_attempt(20), wait=wait_fixed(30))
def get_ingredient_image_url(product_code, lang_code):
    """
    Fetches ingredient image URL using the provided schema.
    """
    url = f"https://world.openfoodfacts.org/api/v2/product/{product_code}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if (
            data["status"] == 1
            and "product" in data
            and "selected_images" in data["product"]
        ):
            images = data["product"]["selected_images"]
            if (
                "ingredients" in images
                and lang_code in images["ingredients"]["display"]
            ):
                return images["ingredients"]["display"][
                    lang_code
                ]  # Access using the language code directly
            else:
                return None
        else:
            return None  # Problem with API response or data

    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        logger.warning(
            f"Error fetching image data for {product_code} ({lang_code}): {e}"
        )
        raise e


def enrich_dataset_with_images(
    dataset_path="openfoodfacts/spellcheck-benchmark",
    data_file="data/train-00000-of-00001.parquet",
):
    """
    Downloads the dataset, retrieves image URLs using .apply(), and adds them to a DataFrame.
    """
    try:
        parquet_file = hf_hub_download(
            repo_id=dataset_path, repo_type="dataset", filename=data_file
        )
        df = pd.read_parquet(parquet_file)

        # Use .apply() to fetch image URLs in parallel
        df["image_url"] = df.apply(
            lambda row: get_ingredient_image_url(row["code"], row["lang"]), axis=1
        )
        # df['image_url2'] = df["code"].apply(get_product_image_url)
        #
        # df['image_url'] = df['image_url2'].combine_first(df['image_url1'])

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_ingredients(df: pd.DataFrame):
    llm = LlmOCR(llm_client=llm_google_client)

    df["predicted"] = df.image_url.apply(lambda x: llm.extract(image_url=x).ingredients)

    return df


if __name__ == "__main__":
    # # Example usage:
    # _product_id = "3168930010883"
    # _image_url = get_ingredient_image_url(_product_id, "fr")
    #
    # print(f"Image URL for product {_product_id}: {_image_url}")

    df_with_images = enrich_dataset_with_images().head(10)  #  TODO REMOVE

    print(df_with_images["image_url"])
    print(df_with_images["image_url"].isna().mean())

    df_with_images = df_with_images.dropna(subset="image_url")

    df_with_images = extract_ingredients(df_with_images)

    for code, lang, reference, hypothesis, image_url in zip(
        df_with_images.code,
        df_with_images.lang,
        df_with_images.reference,
        df_with_images.predicted,
        df_with_images.image_url,
    ):
        print(code)
        print(f"https://world.openfoodfacts.org/api/v2/product/{code}.json")
        print(lang)
        print(reference)
        print(hypothesis)
        print(image_url)
        print("   ")

    word_error_rate = mean(
        [
            wer(reference=reference, hypothesis=hypothesis)
            for reference, hypothesis in zip(
                df_with_images.reference, df_with_images.predicted
            )
        ]
    )

    print(f"{word_error_rate=}")

    char_error_rate = mean(
        [
            cer(reference=reference, hypothesis=hypothesis)
            for reference, hypothesis in zip(
                df_with_images.reference, df_with_images.predicted
            )
        ]
    )

    print(f"{char_error_rate=}")
