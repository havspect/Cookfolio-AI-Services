import base64
import os
import time
from typing import List

from pydantic import ValidationError
import requests
from models import Recipe, RecipeInstructionList
from llm import MODEL_NAME
from litellm import completion

from recipe_scrapers import scrape_me, SCRAPERS
import ingredient_slicer
import langdetect
import deepl

import litellm

def get_recipe_from_image(image_path: str) -> Recipe:
    """
    Extracts recipe information from an image using an LLM.
    
    Args:
        image_path (str): Path to the image containing the recipe.
    
    Returns:
        Recipe: A Recipe object containing the extracted information.
    """
    litellm.enable_json_schema_validation = True

    assert litellm.supports_vision(model=MODEL_NAME) == True

    # Read and encode the image as base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    print(f"Image {image_path} encoded to base64.")
    # Determine MIME type based on file extension
    mime_type = "image/jpeg"
    if image_path.lower().endswith('.png'):
        mime_type = "image/png"
    elif image_path.lower().endswith('.webp'):
        mime_type = "image/webp"

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant that extracts recipes from images.'},
        {
            'role': 'user', 
            'content': [
                {
                    'type': 'text',
                    'text': 'Please extract the recipe from the following image:'
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:{mime_type};base64,{base64_image}',
                        'format': mime_type
                    }
                }
            ]
        }
    ]

    response = completion(
        model=MODEL_NAME,
        messages=messages,
        response_format=Recipe
    )

    try:
        recipe = Recipe.model_validate_json(response.choices[0].message.content)

        return recipe
    except ValidationError as e:
        print(f"Validation error: {e}")
        return None


def get_recipe_from_url(recipe_url: str) -> Recipe:

    # Extract base url and compare against recipe_scrapers supported domains
    base_url = recipe_url.split('/')[2].replace('www.', '')

    if base_url not in SCRAPERS.keys():
        raise ValueError(f"Unsupported URL: {base_url}. Supported domains are: {', '.join(SCRAPERS.keys())}")

    scraper = scrape_me(recipe_url)

    resp = completion(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant that splits recipe instructions into distinct steps.'},
            {'role': 'user', 'content': f'Please format the following recipe instructions: {scraper.instructions()}'}
        ],
        response_format=RecipeInstructionList
    )

    recipe_instructions = RecipeInstructionList.model_validate_json(resp.choices[0].message.content)

    deepl_client = deepl.DeepLClient(os.environ.get("DEEPL_AUTH_KEY"))

    recipe_ingredients = []

    for ingredient in scraper.ingredients():
        if langdetect.detect(ingredient) != 'en':
            ingredient_en = deepl_client.translate_text(ingredient, target_lang='EN-US').text
        
        try:
            sliced_ing = ingredient_slicer.IngredientSlicer(ingredient_en)
            recipe_ingredients.append({
                "name": sliced_ing.food(),
                "quantity": sliced_ing.quantity(),
                "unit": sliced_ing.standardized_unit()
            })
        except:
            recipe_ingredients.append({
                "name": ingredient_en,
                "quantity": "",
                "unit": ""
            })


    sliced_ingredients = [ingredient_slicer.IngredientSlicer(ing) for ing in scraper.ingredients()]

    # Create a Recipe object from the scraper data
    recipe = Recipe(
        title=scraper.title(),
        description=scraper.description(),
        ingredients=recipe_ingredients,
        instructions=recipe_instructions.instructions
    )

    return recipe


def generate_image_for_recipe(recipe: Recipe) -> str:
    """
    Generates an image for a given recipe using an LLM.
    
    Args:
        recipe (Recipe): The recipe for which to generate an image.
    
    Returns:
        str: Base64 encoded image of the generated recipe.
    """

    messages = [
        {'role': 'system', 'content': """
         You are an expert image prompt engineer specializing in vibrant, abstract, isometric art.

        Given a recipe description, your task is to generate a highly detailed image prompt that visually translates the unique ingredients, textures, and mood of the recipe into an abstract, colorful, isometric illustration.

        Instructions:
        - Analyze the recipe and select visually distinctive elements (e.g., main ingredients, textures, preparation methods, or cultural associations).
        - Map these elements into abstract visual motifs, geometric forms, or color palettes.
        - Specify color schemes and isometric composition.
        - Use vivid, descriptive language; avoid generic phrases or placeholders.
        - Maintain the style: abstract, colorful, isometric, with a strong Pop Art/cubism influence, intricate details, flat color, strong lines, and a digital look.
        - Return only the prompt text. Do not include any explanations, introductions, or additional information.

        Example for a cat in this style:
        “Abstract expressionist Pop Art, cubist geometry, floating cat face (no body), bold green and orange, intricate patterns, holographic effect, black background, vibrant digital illustration, flat color, 2D, strong lines, isometric perspective.”"""},
        {'role': 'user', 'content': f'Now, generate a similarly specific and vivid prompt for an image that represents the following recipe description: {recipe.title}. Description: {recipe.description}. Ingredients: {", ".join([f"{ing.name}" for ing in recipe.ingredients])}'}
    ]

    resp = completion(
        model=MODEL_NAME,
        messages=messages,
    )

    image_prompt = resp.choices[0].message.content
    print(f"Generated image prompt: {image_prompt}")

    request = requests.post(
        'https://api.bfl.ai/v1/flux-kontext-pro',
        headers={
            'accept': 'application/json',
            'x-key': os.environ.get("BFL_API_KEY"),
            'Content-Type': 'application/json',
        },
        json={
            'prompt': image_prompt,
        },
    ).json()

    request_id = request["id"]
    polling_url = request["polling_url"] # Use this URL for polling


    while True:
        time.sleep(0.5)
        result = requests.get(
            polling_url,
            headers={
                'accept': 'application/json',
                'x-key': os.environ.get("BFL_API_KEY"),
            },
            params={'id': request_id}
        ).json()
        
        if result['status'] == 'Ready':
            print(f"Image ready: {result['result']['sample']}")
            break
        elif result['status'] in ['Error', 'Failed']:
            print(f"Generation failed: {result}")
            break

    return result['result']['sample']  

recipe = get_recipe_from_image("../data/1000046423.jpg")
recipe_img_url = generate_image_for_recipe(recipe)

# recipe = get_recipe_from_url("https://www.chefkoch.de/rezepte/1943071316420669/Kohlrabigemuese-mit-heller-Sauce.html")
# print(recipe)
