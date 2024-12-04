## ingredient_extraction

This repository provides a tool for extracting ingredients from recipes, specifically focusing on images containing ingredient lists.

### Installation

**1. Create a Conda Environment:**

```bash
conda create -n ingredient_extraction python=3.11
conda activate ingredient_extraction
```

**2. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**3. GEMINI api key:**

**Setting Up `GOOGLE_API_KEY` Using a `.env` File**

* Create a `.env` File:
   - Create a file named `.env` in your project's root directory.
   - This file should not be committed to your version control system (like Git) to keep your API key secure.


* Add Your API Key:
   - Open the `.env` file and add the following line, replacing `YOUR_API_KEY` with your actual GEMINI API key:
     ```
     GOOGLE_API_KEY=YOUR_API_KEY
     ```

### Usage

This library utilizes a Large Language Model (LLM) to analyze images and identify ingredients.  Here's an example using Google's LLM client:

```python
from pydantic import BaseModel

from ingredient_extraction.llm_clients import llm_google_client
from ingredient_extraction.llm_ocr import LlmOCR

# Initialize the OCR client with Google LLM
llm = LlmOCR(llm_client=llm_google_client)

# Define the image URL containing an ingredient list
url = "https://images.openfoodfacts.org/images/products/316/893/001/0883/ingredients_fr.7.400.jpg"

# Extract ingredients from the image
result = llm.extract(image_url=url)

# Assert the extracted data is a BaseModel object
assert isinstance(result, BaseModel)

# Assert "Avoine complète (32%)" is present in the extracted ingredients list
assert "Avoine complète (32%)" in result.ingredients
```
