# test_translate.py

import asyncio
from googletrans import Translator

def translate_text(text, src_lang, dest_lang):
    # Initialize translator (using the more stable Google API endpoint)
    translator = Translator(service_urls=["translate.googleapis.com"])
    # Translate text
    result = translator.translate(text, src=src_lang, dest=dest_lang)
    return result.text

async def main():
    # Initialize translator (using the more stable Google API endpoint)
    translator = Translator(service_urls=["translate.googleapis.com"])

    text = "Hello, world!"
    # Translate from English to Thai
    result = await translator.translate(text, src="en", dest="vi")
    result = await translator.translate(text, src="en", dest="zh")

    print(f"Original:   {text}")
    print(f"Translated: {result.text}")


if __name__ == "__main__":
    import time

    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
