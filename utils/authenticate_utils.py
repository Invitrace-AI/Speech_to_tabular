#External package
import openai

#Built-in package
import os
from dotenv import load_dotenv

def authenticate():
    try:
        openai.api_key = os.environ['OPENAI-API-KEY']

    # Design for local run
    except KeyError:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('OPENAI-API-KEY')
        openai.api_key = api_key