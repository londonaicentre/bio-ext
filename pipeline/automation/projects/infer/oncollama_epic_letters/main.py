import json
import re
from typing import Iterable

import requests

import vllm

"""
main.py
Functions required to pass Epic cancer document content to an LLM endpoint, and parse/save the output json.
Currently loops through files in a directory, and saves to a target directory. Todo:
(1) Modify to read from Elastic, then save back to new Elastic index.
(2) Need to spin model up and down if being run on an increment through Dagster
    - Or could 'orchestrate' via shell script for now?
(3) Need to choose what metadata to keep in target json.
"""


def extract_epic_section(text: str) -> str | None:
    """
    Given an Epic cancer document, extract relevant section only (strip headers) if starting word/phrase exists
    """
    start_idx = -1
    text_lower = text.lower()

    if "diagnosis" in text_lower:
        start_idx = text_lower.index("diagnosis")
    elif "diagnoses" in text_lower:
        start_idx = text_lower.index("diagnoses")
    elif "mdm" in text_lower:
        start_idx = text_lower.index("mdm")
    elif "mdt" in text_lower:
        start_idx = text_lower.index("mdt")
    elif "multidisciplinary" in text_lower:
        start_idx = text_lower.index("multidisciplinary")
    elif "trial clinic" in text_lower:
        start_idx = text_lower.index("trial clinic")
    elif "trial unit" in text_lower:
        start_idx = text_lower.index("trial unit")
    elif "trials clinic" in text_lower:
        start_idx = text_lower.index("trials clinic")
    elif "trials unit" in text_lower:
        start_idx = text_lower.index("trials unit")
    elif "clinical trial" in text_lower:
        start_idx = text_lower.index("clinical trial")
    elif "early phase" in text_lower:
        start_idx = text_lower.index("early phase")
    elif "trial" in text_lower:
        start_idx = text_lower.index("trial")
    elif "referring" in text_lower:
        start_idx = text_lower.index("referring")
    elif "seen in" in text_lower:
        start_idx = text_lower.index("seen in")
    else:
        return None

    return text[start_idx:]


def extract_json_from_text(text: str) -> dict:
    """
    Repeatedly tries to parse the first JSON object found in the text, and deserialises it into a Python dictionary.
    If the text is not valid JSON, it will try to find a JSON-like structure within the text and parse it.

    To be used to handle the output from the LLM, which may not be valid JSON.

    Args:
        text (str): The input text to extract JSON
    Returns:
        dict: The extracted JSON object
    Raises:
        ValueError: If no valid JSON object is found in the text
    """
    # try direct pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        # findg anything that might look like a json
        pattern = r"({[\s\S]*}|\[[\s\S]*\])"
        matches = re.finditer(pattern, text)

        # find the longest match
        potential_jsons = [(match.group(0), len(match.group(0))) for match in matches]
        potential_jsons.sort(key=lambda x: x[1], reverse=True)

        for potential_json, _ in potential_jsons:
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError:
                continue

        raise ValueError(
            "No valid JSON object found in the text. Please check the input text format."
        )
    except Exception as e:
        print(f"Error searching for JSON object: {e}")
        raise ValueError(
            "Failed to extract JSON object from text. Please check the input text format."
        ) from e

from projects.infer.config import OncollamaConfig
from vllm.entrypoints.llm import LLM
from vllm import SamplingParams

def process_locally(documents: Iterable[str], oncollama_config: OncollamaConfig) -> None:
    model = LLM(model=oncollama_config.model_path)

    sampling_params = SamplingParams(temperature=oncollama_config.temperature)

    for document in documents:
        # Remove unwanted parts of the input string retrieved from Elastic
        focused_document = extract_epic_section(document)

        if focused_document is None:
            print("No relevant section found in document: skipping")
            continue

        # Process the document via vLLM

        response = model.generate(focused_document, sampling_params=sampling_params)
        response_content = response[0].text

        # Extract JSON from response
        try:
            extracted_json = extract_json_from_text(response_content)
        except ValueError as e:
            print(f"Error extracting JSON from response: {e}")
            continue

        # Handle the extracted JSON as needed
        print(extracted_json)



def process_document(
    input_string: str, api_url: str = "http://localhost:8080/v1/chat/completions"
) -> dict:
    """
    Processes a document and handles the response from the LLM.

    Args:
        input_string (str): The document content to be processed.
        api_url (str): The URL of the LLM API endpoint.
    Returns:
        str: The extracted JSON object as a string.
    Raises:
        ConnectionError: If there is a connection error while sending the request to the LLM API.
        HTTPError: If the response from the LLM API is not successful.
        ValueError: If no relevant section is found in the document or if JSON extraction fails.
    """
    # Remove unwanted parts of the input string retrieved from Elastic
    focused_document = extract_epic_section(input_string)

    if focused_document is None:
        raise ValueError("No relevant section found in document: skipping")

    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "oncollamav2",
        "messages": [
            {"role": "system", "content": },
            {"role": "user", "content": focused_document},
        ],
        "temperature": 0.0,
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        # parse out contents section
        response_content = response.json()["choices"][0]["message"]["content"]
    # We probably shouldn't be catching exceptions at all - but let the error bubble up immediately to the caller to handle.
    # If any of these were to occur we would probably want to stop the pipeline and alert the user.
    except requests.ConnectionError as e:
        print(f"Error connecting to LLM endpoint: {e}")
        raise e
    except requests.HTTPError as e:
        print(f"HTTP error: {e}")
        raise e

    # extract JSON from response
    try:
        extracted_json = extract_json_from_text(response_content)
    except ValueError as e:
        print(f"Error extracting JSON from response: {e}")
        raise e

    # Return the extracted JSON
    return extracted_json
