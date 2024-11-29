from pydantic import BaseModel


class ParsingOutput(BaseModel):
    ingredients: str


if __name__ == "__main__":
    categories = ParsingOutput(ingredients="Salt")

    print(categories)
