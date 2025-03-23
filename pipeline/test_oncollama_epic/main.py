import json
import os
import re
import requests
from pathlib import Path

"""
main.py
Functions required to pass Epic cancer document content to an LLM endpoint, and parse/save the output json.
(1) Source json loaded
(2) Try to extract target section
(3) Pass to LLM
(4) Extract section, check for json, create output json
(5) Save output json
Currently loops through files in a directory, and saves to a target directory. Todo:
(1) Modify to read from Elastic, then save back to new Elastic index.
(2) Need to spin model up and down if being run on an increment through Dagster
    - Or could 'orchestrate' via shell script for now?
(3) Need to choose what metadata to keep in target json.
"""

def extract_epic_section(text):
    """
    Given an Epic cancer document, extract relevant section only (strip headers) if starting word/phrase exists
    """
    start_idx = -1
    text_lower = text.lower()

    if 'diagnosis' in text_lower:
        start_idx = text_lower.index('diagnosis')
    elif 'diagnoses' in text_lower:
        start_idx = text_lower.index('diagnoses')
    elif 'mdm' in text_lower:
        start_idx = text_lower.index('mdm')
    elif 'mdt' in text_lower:
        start_idx = text_lower.index('mdt')
    elif 'multidisciplinary' in text_lower:
        start_idx = text_lower.index('multidisciplinary')
    elif 'trial clinic' in text_lower:
        start_idx = text_lower.index('trial clinic')
    elif 'trial unit' in text_lower:
        start_idx = text_lower.index('trial unit')
    elif 'trials clinic' in text_lower:
        start_idx = text_lower.index('trials clinic')
    elif 'trials unit' in text_lower:
        start_idx = text_lower.index('trials unit')
    elif 'clinical trial' in text_lower:
        start_idx = text_lower.index('clinical trial')
    elif 'early phase' in text_lower:
        start_idx = text_lower.index('early phase')
    elif 'trial' in text_lower:
        start_idx = text_lower.index('trial')
    elif 'referring' in text_lower:
        start_idx = text_lower.index('referring')
    elif 'seen in' in text_lower:
        start_idx = text_lower.index('seen in')
    else:
        return None

    return text[start_idx:]

def extract_json_from_text(text):
    """
    Aggressively extract broadest JSON object from any text
    """
    # try direct pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        # findg anything that might look like a json
        pattern = r'({[\s\S]*}|\[[\s\S]*\])'
        matches = re.finditer(pattern, text)

        # find the longest match
        potential_jsons = [(match.group(0), len(match.group(0))) for match in matches]
        potential_jsons.sort(key=lambda x: x[1], reverse=True)

        for potential_json, _ in potential_jsons:
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError:
                continue

        return None
    except Exception as e:
        print(f"Error searching for JSON object: {e}")
        return None

def get_system_prompt():
    """
    Returns the system prompt for the oncology extraction model
    """
    return """
    You are an oncology specialist extracting cancer information into a structured schema. Your task is to analyze medical documents and produce accurate structured data according to the schema below. Precision of data extraction is vital, as this is part of a medico-legal process, and inaccuracies could lead to harm.\n\nOUTPUT SCHEMA:\n{\"metadata\":{\"version\":\"2.2\",\"schema_guidelines\":[\"Extract close to original text. Preserve original clinical language and abbreviations, do not standardise\",\"Extract dates as year\/month ONLY where available and clearly attached to concept\",\"Do not include subfields where information is missing, instead, present empty blocks. Do not infer or make up new information\",\"Duplicative extraction is allowed - you can place the same item into multiple fields IF this is appropriate\"]},\"clinical_content\":{\"primary_cancer\":{\"description\":\"This inclusion group contains facts and a timeline relevant to the primary cancer that is the main subject of letter. Use fields only where information present.\",\"repeatable\":false,\"primary_cancer_facts\":{\"description\":\"This inclusion group contains facts about the primary cancer that is the main subject of letter.\",\"repeatable\":false,\"fields\":{\"site\":\"type:string. Confirmed main primary organ site or topography or haemoatological type (e.g. - ovary, breast, brain, lung, unknown origin, non-hodgkin's, diffuse large b-cell lymphoma etc) and more detailed localisation (e.g. right, upper lobe, cerebellum). Confirmed diagnosis only. For example - do not populate if patient is referred for suspicion of diagnosis and further invesigation\",\"year\":\"type:YYYY. Year of initial diagnosis (YYYY), if given\",\"month\":\"type:MM. Month of initial diagnosis as 1 to 12, if given\",\"performance_status\":\"type:numeric. The most recent patient performance status score given in the text\",\"performance_status_scale\":\"type:string. What performance status scale is used, e.g. ECOG, KPS\",\"tnm_stage\":\"type:string. Current TNM staging for main cancer. Do NOT infer.\",\"other_stage\":\"type:string. Other cancer staging, (e.g. stage 1, stage III etc). Do NOT infer.\",\"histopathology_status\":\"type:string. Morphology (e.g. if adenocarcinoma, squamous cell carcinoma, small cell carcinoma, lymphoma, leukaemia etc etc)\",\"histopathology_grade\":\"type:string. Any grading criteria, e.g. Grade 2, or Gleason or FIGO grading etc\",\"spread_block\":{\"description\":\"Block describing set of locations of confirmed disease spread, including nodal + metastatic disease. Repeat as needed to cover different locations\",\"repeatable\":true,\"fields\":{\"site\":\"type:string. Name of best description for specific metastastic site or nodal involvement, e.g. lung, liver, sentinel lymph node etc.\"}},\"spread_desc\":\"type:string. Any additional, general descriptions of disease spread, e.g. widespread, diffusely metastatic\",\"biomarker_block\":{\"description\":\"Block describing cancer biomarkers, including genomic, laboratory, histopathological, etc. Repeat as needed to cover different biomarkers\",\"repeatable\":true,\"fields\":{\"biomarker_name\":\"type:string. Name of single biomarker of interest attached to the primary diagnosis, including variants (E.g. ER, PR, JAK2 with V617, BRCA1, Microsatellite instability etc)\",\"binary_status\":\"type:enum['+ve', '-ve']. The binary status of the biomarker for this patient (e.g. where ER is +ve or -ve)\",\"status_desc\":\"type:string. Descriptive or numerical status of biomarker, often the level of positivity (e.g. strongly positive, 60% percent cells, or copy number variation, or 3+ positivity).\"}}}},\"cancer_timeline\":{\"description\":[\"Timeline of key events that have happened to the patient for main primary cancer. Include any events out of key types up to and including the current consultation. Populate only with allowed types, disregard other types of events. For each event, populate with rich descriptive text that fully describes the event in question, including any scoring systems (e.g. RECIST). Note that the same text is allowed to appear in multiple fields, for example 'CT scan shows tumor size increase to 3.6cm' would appear as both 'got_radiology_result', and 'experienced_cancer_disease_progression'.\"],\"repeatable\":true,\"allowed_types\":[\"started_new_systemic_treatment\",\"started_new_radiotherapy_treatment\",\"had_surgical_treatment_performed\",\"experienced_treatment_toxicity_or_complication\",\"experienced_change_or_stop_to_existing_treatment\",\"enrolled_to_clinical_trial\",\"withdrawn_from_clinical_trial\",\"got_radiology_result\",\"got_pathology_result\",\"got_laboratory_result\",\"experienced_positive_treatment_response_or_improvement\",\"experienced_cancer_disease_stability\",\"experienced_cancer_disease_progression\",\"experienced_general_deterioration\",\"patient_died\"],\"fields\":{\"type\":\"type:enum from allowed_types. Describe the type of event.\",\"value\":\"type:string. Extracted descriptive text decribing event, remaining close to original text\",\"year\":\"type:YYYY. Year of event (YYYY) if given\",\"month\":\"type:MM. Month of event (1-12) if given\"}}},\"other_cancers\":{\"description\":\"Sometimes patients have had other cancers. This contains array of historical cancer diagnoses and facts, if any are mentioned. Only minimal information is collected for historical cancers.\",\"repeatable\":true,\"fields\":{\"site\":\"type:string. Organ site or topography, and more detailed localisation\",\"year\":\"type:YYYY. Year of historical cancer diagnosis (YYYY) if given\",\"month\":\"type:MM. Month of historical diagnosis (1-12) if given\",\"spread_all\":\"type:string. List the location(s) of confirmed spread, including mets and nodal disease. Positive mentions only.\",\"tnm_stage\":\"type:string. TNM staging for historical cancer\",\"other_stage\":\"type:string. Other cancer staging\",\"histopathology_status\":\"type:string. Histopathological classification, morphology, and findings for historical cancer\",\"histopathology_grade\":\"type:string. Any grading criteria, e.g. Grade 2, or Gleason or FIGO grading etc\",\"biomarkers_all\":\"type:string. List any identifying biomarkers for historical cancer, including genomic and pathological biomarkers\",\"latest_situation\":\"type:string. Describe the current status for this cancer, (e.g.- in remission, or under active surveillance, receiving systemic treamtent, etc)\"}},\"patient_findings\":{\"description\":\"Active patient information given in the letter uncovered during a consultation. Comorbidities are concurrent long term conditions. Symptoms are self-reported. Physical findings are from examination or observation, functonal findings are about patients functional status, mental state and qol are about patient's mental feelings and their quality of life.\",\"repeatable\":true,\"allowed_types\":[\"comorbidity_finding\",\"symptom_finding\",\"physical_finding\",\"functional_finding\",\"mental_state_or_qol_finding\"],\"fields\":{\"type\":\"type:enum from allowed_types. Type of patient finding\",\"value\":\"type:string. Extracted descriptive text for the finding, remain close to original text\"}},\"clinical_summary\":{\"description\":[\"This section gives a one-line factual summary of most up to date cancer and treatment status, and a one-line clinical expert impression of how the patient is doing - as if you were the document writer. If this information is NOT present, please state so here.\"],\"repeatable\":false,\"fields\":{\"summary\":\"type:string. Summarise the most up to date cancer and treatment status, and summarise the overall clinical impression, or say that this information is missing.\"}},\"future_plan\":{\"description\":[\"This final section gives next steps, often laid out as part of a plan.\"],\"repeatable\":true,\"allowed_types\":[\"planned_treatment\",\"planned_investigation\",\"planned_referral_to_other_team\",\"planned_other\"],\"fields\":{\"type\":\"type:enum from allowed_types. Type of next step from allowed_types\",\"value\":\"Extracted information, staying close to original text\"}}}}.\n\nEXTRACTION GUIDELINES:\n1. Extract ONLY explicitly mentioned information and preserve original clinical terminology\n2. Include dates (year/month) only when clearly stated\n3. Omit fields when information is absent - return empty blocks rather than inferring data\n4. The same information can appear in multiple relevant fields\n5. For timeline and events, patient findings, and future plans, use ONLY the allowed_types given in the schema\n\nSPECIAL CASES:\n1. Do NOT populate primary_cancer fields for suspected but unconfirmed diagnoses\n2.For empty/irrelevant documents, return empty schema with summary: \"The provided content is not related to cancer, and no content could be extracted\".FINAL CHECK: Double check that all appropriate cancer facts are all extracted, and that all timeline components have been extracted\n\n FINAL OUTPUT:\nReturn ONLY the valid JSON object according to this schema without commentary or additional text or repeating the document, like this: {\"primary_cancer\":{...},\"other_cancers\":[...],\"patient_findings\":[...],\"clinical_summary\":{...},\"future_plan\":[...]}.
    """

def process_document_content(document_content, api_url="http://localhost:8080/v1/chat/completions"):
    """
    Process document content through the fine-tuned LLM
    """

    text_to_process = extract_epic_section(document_content)

    if text_to_process is None:
        print("No relevant section found in document: skipping")
        return None

    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "oncollamav2",
        "messages": [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": text_to_process},
        ],
        "temperature": 0.0,
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        # parse out contents section
        response_content = response.json()["choices"][0]["message"]["content"]

        # extract JSON from response
        extracted_json = extract_json_from_text(response_content)

        return extracted_json
    except Exception as e:
        print(f"Error processing document: {e}")
        return None

def read_json_file(file_path):
    """
    Read and parse a JSON file
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def process_single_file(file_path, output_dir):
    """
    Process a single JSON file and save the result
    """
    input_data = read_json_file(file_path)
    if not input_data:
        return False

    try:
        document_content = input_data["_source"]["document_Content"]

        extraction_result = process_document_content(document_content)

        if extraction_result is None:
            print(f"Skipping {file_path} - unable to extract relevant information")
            return False

        # create output json with original metadata
        if extraction_result:
            output_data = {
                "source_id": input_data.get("_id"),
                "patient_SourceId": input_data["_source"].get("patient_SourceId"),
                "patient_EpicId": input_data["_source"].get("patient_EpicId"),
                "patient_NhsNumber": input_data["_source"].get("patient_NhsNumber"),
                "activity_Department": input_data["_source"].get("activity_Department"),
                "source_document": input_data["_source"].get("document_Content"),                
                "extraction_result": extraction_result
            }

            output_filename = f"processed_{os.path.basename(file_path)}"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"Success: {file_path}")
            return True

        else:
            print(f"Failed to extract data from: {file_path}")
            return False

    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def process_directory(input_dir, output_dir):
    """
    Process all JSON files in a directory
    For testing purposes only
    """
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    # loop through files
    success_count = 0
    for file_path in json_files:
        if process_single_file(file_path, output_dir):
            success_count += 1

    print(f"Successfully processed {success_count} files.")

def main():
    input_dir = "testdata"
    output_dir = "testoutputs"

    os.makedirs(output_dir, exist_ok=True)

    process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()