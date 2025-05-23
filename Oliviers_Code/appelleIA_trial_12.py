import requests
from tqdm import tqdm
import os
import fitz
import time
from datetime import timedelta


def pdf_to_str(pdf_path):
    with fitz.open(pdf_path) as doc:
        full_text = ""
        for page in doc:
            full_text += page.get_text()
    
    return full_text

studies = [
    "AI - Proof of concept (Study 1).pdf",
    #"AI - Proof of concept (Study 2).pdf", #To remove, because it is the same study that 3, and the model is not yet able to distinguish
    #"AI - Proof of concept (Study 3).pdf", #To remove, because it is the same study that 2, and the model is not yet able to distinguish
    "AI - Proof of concept (Study 4).pdf",
    "AI - Proof of concept (Study 5).pdf",
    #"AI - Proof of concept (Study 6).pdf", # To remove because no OCR
    "AI - Proof of concept (Study 7).pdf",
    "AI - Proof of concept (Study 8).pdf",
    "AI - Proof of concept (Study 9).pdf",
    #"AI - Proof of concept (Study 10).pdf", # To remove because no OCR
    "AI - Proof of concept (Study 11).pdf",
    "AI - Proof of concept (Study 12).pdf"
           ]

contexts = []

for studie in studies :
    context = pdf_to_str(studie)
    contexts.append(context)

# Liste des questions à poser
questions = [
    '''
    Classify the exposure type in this study as one of the following

    Dermal: involves the application of the substance directly onto the animal’s skin using a patch or a wrap. It can also be applied topically, or injected intravenously or intradermally.
    Oral: involved feeding the animal with amounts of the substance through their mouth. Can be by gavage or by putting substance in the animal’s feed.
    Inhalation: involves administration of the substance in vapour or powder or dust form. The animal can be in contact with the substance through only its nose in powder form or vapour form or through its whole body in vapour form by being in a gas chambre.
    Null: the study doesn’t use live animals/is in-vitro or there is no exposure type mentioned.
    When you’ve determined which exposure type this study uses, your final line must follow this format exactly: EXPOSURE TYPE – [type] or EXPOSURE TYPE – null if there was no exposure type.
    ''',
    '''
    Classify the exposure type in this study as one of the following using its keywords

    Dermal: wrap, patch, occlusive, intradermal, topical, paper, epidermal, intravenous, inulladermal, injection, irritation, repeated insult
    Oral: involved feeding the animal with amounts of the substance through their mouth. Can be by gavage or by putting substance in the animal’s feed
    Inhalation: nose, dust, vapour, vapor, respirable, breathing zone, room air, chamber
    Null: the study doesn’t use live animals/is in-vitro or there is no exposure type mentioned.
    When you’ve determined which exposure type this study uses, your final line must follow this format exactly: EXPOSURE TYPE – [type] or EXPOSURE TYPE – null if there was no exposure type.
    ''',
    '''
    Read the study and assign it one exposure type of the next three by process of elimination (which two exposure types do not apply to this study based on their definition).

    DERMAL: involves the application of the substance directly onto the animal’s skin using a patch or a wrap. It can also be applied topically, or injected intravenously or intradermally.
    ORAL: involved feeding the animal with amounts of the substance through their mouth. Can be by gavage or by putting substance in the animal’s feed.
    INHALATION: involves administration of the substance in vapour or powder or dust.
    If there was no exposure type, which is possible when studies are in-vitro or when the type is simply not mentioned, then the answer is NULL.

    When you’ve determined which exposure type this study uses, your final line must follow this format exactly: EXPOSURE TYPE – [type] or EXPOSURE TYPE – null.
    ''',
    '''
    Was there any information about the purity of the compound tested? If yes, the final word of your answer should be 'PURITY : [purity]%’. If not, the final word of your answer should be 'PURITY : No info'
    ''',
    '''
    Was there any information about the purity of the compound tested? Information on purity is a percentage and is often in a table in the row or column of purity or batch purity. If yes, the final word of your answer should be 'PURITY : [purity]%'. If no, the final word of your answer should be 'PURITY : No info'
    ''',
    '''
    Find the purity of the tested substance. The final word of your answer should be 'PURITY : [purity]%’ or ‘PURITY : no info’ if the purity isn’t mentioned.
    ''',
    '''
    Was there any information about the vehicle solvent or vehicle article used? If yes, the final word of your answer should be ‘SOLVENT : [solvent]. If not, the final word of your answer should be ‘SOLVENT : no info’ .
    ''',
    '''
    Was there any information about the vehicle solvent or vehicle article used? Vehicles are other non-toxic substances that are used to carry the substance that is being analyzed for toxicity. Often, the information is in a table in the row or column named vehicle or next to the word “vehicle”. If there is no table, the vehicle is often mentioned as a mixture with the substance being analyzed in order to dilute it. If yes, the final word of your answer should be ‘SOLVENT : [solvent]. If not, the final word of your answer should be ‘SOLVENT : no info’ .
    ''',
    '''
    Was there any mention of vehicle solvents? Vehicle solvents are most often different types of alcohol, water, methanol, DMSO, different types of oil, aqueous methylcellulose, acetone, petrolatum, sodium chloride, gelatin capsule. The vehicle solvent is often mentioned in a table or mentioned by saying the studied substance will be diluted with it. By using these keywords and the context of the study, find the vehicle solvent. The final word of your answer should be ‘SOLVENT : [solvent]. If not, the final word of your answer should be ‘SOLVENT : no info’ .
    '''
]


# === Base URL for Ollama chat
url = "http://localhost:11434/api/chat"

# === Iterate over each context
for run_index in range(2, 4):  # Run1 to Run3
    run_folder = os.path.join("conversations", f"Run{run_index}")
    os.makedirs(run_folder, exist_ok=True)
    print(f"\n=== Starting Run {run_index} ===")
    
    timings = []
    for context_index, context in enumerate(contexts, 1):
        folder_name = studies[context_index-1]
        output_dir = os.path.join(run_folder, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nProcessing {folder_name} in Run{run_index}")
        start_time = time.time()

        for i, question in enumerate(tqdm(questions, desc=folder_name, leave=False)):
            # Step 1: Send context
            context_data = {
                "model": "deepseek-r1:14b",
                "messages": [
                    {"role": "user", "content": context}
                ],
                "stream": False
            }

            try:
                context_response = requests.post(url, json=context_data)
                if context_response.status_code == 200:
                    context_reply = context_response.json()["message"]["content"]

                    # Step 2: Ask the question in the same conversation
                    question_data = {
                        "model": "deepseek-r1:14b",
                        "messages": [
                            {"role": "user", "content": context},
                            {"role": "assistant", "content": context_reply},
                            {"role": "user", "content": question}
                        ],
                        "stream": False
                    }

                    question_response = requests.post(url, json=question_data)
                    if question_response.status_code == 200:
                        question_reply = question_response.json()["message"]["content"]

                        filename = os.path.join(output_dir, f"question_{i+1:02d}.txt")
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write("User (Context)    : " + context + "\n")
                            f.write("Assistant (Reply) : " + context_reply.strip() + "\n\n")
                            f.write("User (Question)   : " + question + "\n")
                            f.write("Assistant (Answer): " + question_reply.strip() + "\n")

                    else:
                        print(f"\nError sending question {i+1}: {question_response.status_code}")
                else:
                    print(f"\nError sending context {context_index}: {context_response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"\nConnection error: {e}")
        end_time = time.time()
        elapsed = end_time - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        timings.append((folder_name, elapsed_str))
        print(f"Finished processing {folder_name} in {elapsed_str}")

    with open(f"processing_timesfRun{run_index}.txt", "w", encoding="utf-8") as f:
        for filename, duration in timings:
            f.write(f"{filename} - {duration}\n")