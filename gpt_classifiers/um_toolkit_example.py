from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import base64

#Sets the current working directory to be the same as the file.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Load environment file for secrets.
try:
    if load_dotenv('.env') is False:
        raise TypeError
except TypeError:
    print('Unable to load .env file.')
    quit()
#Create Azure client
client = AzureOpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  
    api_version=os.environ['API_VERSION'],
    azure_endpoint = os.environ['OPENAI_API_BASE'],
    organization = os.environ['OPENAI_ORGANIZATION']
)

#Create Query
messages=[
        {
                    "role": "system",
                    "content": "You are a meticulous expert in technical document analysis, specializing in distinguishing between data visualizations and engineering schematics."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                            Your task is to perform a highly accurate binary classification of the provided image.

                            Please adhere to the following steps:
                            1.  **Analyze Visual Evidence**: First, examine the image for its core components.
                                - For a **2D plot-like graph**, look for elements like axes (X and Y), data points, trend lines, bars, legends, and titles that represent quantitative data.
                                - For a **Nuclear Reactor Schematic Diagram**, look for representations of physical hardware like pressure vessels, control rods, pumps, pipes, heat exchangers, and other engineering symbols.

                            2.  **Formulate Reasoning**: Based on your analysis, write a brief 'reasoning' statement explaining which set of visual evidence is present and why it leads to your conclusion.

                            3.  **Provide Classification**: Finally, classify the image into one of the two categories (and only the two) below.

                            The categories are:
                            - `2D plot-like graph`
                            - `Nuclear Reactor Schematic Diagram`

                            Your output MUST be a valid JSON object with exactly two keys: "reasoning" and "classification". The value for "classification" must be one of the two exact strings listed above.
                            """
                        },
                        {
                            "type": "image_url", "image_url": {"url": f"https://michiganross.umich.edu/sites/default/files/styles/max_1300x1300/public/images/news-story/butterfly.jpeg"}
                        }
                    ]
                }
    ]

response = client.chat.completions.create(
    model=os.environ['MODEL'],
    messages=messages,
    temperature=0.0,
)
print(response)

#Print response.
print(response.choices[0].message.content)