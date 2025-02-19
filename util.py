import numpy as np
import pandas as pd
import re
import json
from tqdm import tqdm
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
torch.random.manual_seed(0)

df = pd.read_csv("legal_text_classification.csv")
df = df.sample(25, random_state = 21)
df3 = df[['case_text']]
df2 = df[['case_title']]


def extract_case_details(case_title):
    # Extract plaintiff and defendant based on 'v' or 'vs.'
    parties = re.split(r'\s+v(?:s\.)?\s+', case_title, maxsplit=1)
    plaintiff = parties[0].strip() if len(parties) > 1 else None

    # Extract and clean defendant's name
    defendant = None
    docket_number = None
    if len(parties) > 1:
        defendant_raw = parties[1]
        defendant_match = re.split(r'\[|\(|\d{4}', defendant_raw, maxsplit=1)
        defendant = defendant_match[0].strip() if defendant_match else None

        # Optimized extraction of docket number: looks for patterns like [YYYY] ABC1234
        docket_match = re.search(r'\[([0-9]{4})\]\s+[A-Z]{3,5}\s+(\d+)', defendant_raw)
        if docket_match:
            docket_number = docket_match.group(2)  # Extract the docket number

    # Extract case year (year from either [YYYY] or (YYYY))
    case_year = None
    # First check for year in parentheses
    date_match = re.search(r'\((\d{4})\)', case_title)
    if date_match:
        case_year = date_match.group(1)
    # If no match in parentheses, check for year in square brackets
    elif not case_year:
        date_match = re.search(r'\[([0-9]{4})\]', case_title)
        if date_match:
            case_year = date_match.group(1)

    # Extract docket number (the first number after the letters in "ABC1234" format)
    if not docket_number:
        docket_match = re.search(r'[A-Z]{3,5}\s+(\d+)', case_title)
        if docket_match:
            docket_number = docket_match.group(1)  # Extract the docket number

    return {
        "plaintiff": [plaintiff] if plaintiff else None,
        "defendant": [defendant] if defendant else None,
        "docket_number": int(docket_number) if docket_number else None,
        "year": int(case_year)
    }

def batchify(data, batch_size):
    """Split data into smaller batches."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def extract_case_details_batch(batch_texts, pipe, generation_args):
    """Process a batch of case texts using the pipeline."""
    formatted_prompts = [
        [
            {
                "role": "system",
                "content": "You are a helpful paralegal. Extract the plaintiff, defendant, docket number, and year from this text and return them in a JSON format. Do not add any additional output."
            },
            {
                "role": "user",
                "content": (
                    "Task: Extract the plaintiff, defendant, docket number, and year from this text and return them in a JSON format.\n"
                    "Format: return each case with a JSON representation like this and do not add any extra text:\n"
                    "{\n"
                    '    "plaintiff": list[str],\n'
                    '    "defendant": list[str],\n'
                    '    "docket_number": int,\n'
                    '    "year": int\n'
                    "}\n\n"
                    "Instructions: Look for the main case in the case text. It should be the main subject of the text and be brought up the most.\n"
                    "In order to extract the data from the case, follow these tips: The plaintiff and defendant's name should be separated by 'v'. The plaintiff and defendant's names should be the names of a person or a corporation and be comprised of a person's last name or the corporation's name. The year should be contained within brackets or parentheses,\n"
                    "and the docket number should be the first group of digits following the year and a sequence of letters.\n\n"
                    "Example 1:\n"
                    "Input: <Ordinarily that discretion will be exercised so that costs follow the event and are awarded on a party and party basis. A departure from normal practice to award indemnity costs requires some special or unusual feature in the case: Alpine Hardwood (Aust) Pty Ltd v Hardys Pty Ltd (No 2) [2002] FCA 224 ; (2002) 190 ALR 121 at [11] (Weinberg J) citing Colgate Palmolive Co v Cussons Pty Ltd (1993) 46 FCR 225 at 233 (Sheppard J).>\n"
                    "Output:\n"
                    "{\n"
                    '    "plaintiff": ["Alpine Hardwood (Aust) Pty Ltd"],\n'
                    '    "defendant": ["Hardys Pty Ltd"],\n'
                    '    "docket_number": 224,\n'
                    '    "year": 2002\n'
                    "}\n\n"
                    "Example 2:\n"
                    "Input: <In Cox v Journeaux (No 2) [1935] HCA 48 ; (1935) 52 CLR 713 , Dixon J (as his Honour then was) considered the meaning of personal injury or wrong done to the bankrupt within s 63(3) of the Bankruptcy Act 1924-1933 (Cth). His Honour applied Wilson v United Counties Bank Ltd [1920] AC 102 and described (at 721) the relevant test in the following terms: The test appears to be whether the damages or part of them are to be estimated by immediate reference to pain felt by the bankrupt in respect of his mind, body or character and without reference to his rights of property.>\n"
                    "Output:\n"
                    "{\n"
                    '    "plaintiff": ["Wilson"],\n'
                    '    "defendant": ["United Counties Bank Ltd"],\n'
                    '    "docket_number": 102,\n'
                    '    "year": 1920\n'
                    "}\n\n"
                    f"Input_text:\n{text}"
                )
            }
        ]
        for text in batch_texts
    ]


    inputs = [prompt for sublist in formatted_prompts for prompt in sublist]


    outputs = pipe(inputs, **generation_args)

    responses = []
    for output in outputs:
        if isinstance(output, dict) and "generated_text" in output:
            responses.append(output["generated_text"])
        else:
            responses.append({"error": "Unexpected output format", "output": output})

    return responses

def extract_case_summary_batch(batch_texts, pipe, generation_args):
    """Process a batch of case texts to extract summaries."""
    formatted_prompts = [
        [
            {
                "role": "system",
                "content": "You are a helpful paralegal. Summarize the given case text in a concise manner, preserving key legal points and outcomes."
            },
            {
                "role": "user",
                "content": (
                    "Task: Summarize the following case text in 2-3 sentences while maintaining key legal information. Make sure to include the case name, city name, docket number, date, and plaintiff and defendant's information if possible in the summary.\n"
                    "Format: Return only the summary as a plain text string without additional explanations or formatting.\n\n"
                    f"Input_text:\n{text}"
                )
            }
        ]
        for text in batch_texts
    ]

    inputs = [prompt for sublist in formatted_prompts for prompt in sublist]
    outputs = pipe(inputs, **generation_args)

    summaries = []
    for output in outputs:
        if isinstance(output, dict) and "generated_text" in output:
            summaries.append(output["generated_text"])
        else:
            summaries.append("Error: Unexpected output format")

    return summaries


def calculate_accuracy(returned_json, case_texts_df, text_column="case_title"):
    """
    Calculate accuracy by checking if the extracted fields appear in the corresponding case text.

    Parameters:
    - returned_json (list of dicts): The extracted JSON data.
    - case_texts_df (pd.DataFrame): DataFrame containing the original case texts.
    - text_column (str): The column name in the DataFrame that contains case texts.

    Returns:
    - float: Accuracy as a percentage.
    """
    import json

    # Ensure returned_json is in the correct format
    if isinstance(returned_json, str):
        returned_json = json.loads(returned_json)  # Convert string to list of dicts
    
    if isinstance(returned_json, list):
        returned_json = [json.loads(item) if isinstance(item, str) else item for item in returned_json]

    correct_count = 0
    total_cases = len(returned_json)

    # Ensure the text column exists in the DataFrame
    if text_column not in case_texts_df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    for i in range(total_cases):
        extracted = returned_json[i]

        # Debug: Ensure extracted is a dictionary
        if not isinstance(extracted, dict):
            print(f"⚠️ Warning: Entry {i} in returned_json is not a dict: {extracted}")
            continue  # Skip incorrect entries

        case_text = str(case_texts_df.iloc[i][text_column]).lower()  # Convert to lowercase for case-insensitive matching
        
        # Extract fields safely
        plaintiff = extracted.get("plaintiff", []) or []
        defendant = extracted.get("defendant", []) or []
        docket_number = extracted.get("docket_number", None)
        year = extracted.get("year", None)
        # Matching logic
        plaintiff_match = any(name.lower() in case_text for name in plaintiff) if plaintiff else False
        defendant_match = any(name.lower() in case_text for name in defendant) if defendant else False
        docket_match = str(docket_number) in case_text if docket_number else False
        year_match = str(year) in case_text if year else False

        # Count as correct if at least one field matches
        if plaintiff_match or defendant_match or docket_match or year_match:
            correct_count += 1

    accuracy = (correct_count / total_cases) * 100 if total_cases > 0 else 0
    return round(accuracy, 2)
