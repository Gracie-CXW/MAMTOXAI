import requests
from tqdm import tqdm
import os
import fitz


def pdf_to_str(pdf_path):
    # Ouvrir le document PDF
    with fitz.open(pdf_path) as doc:
        full_text = ""
        for page in doc:
            full_text += page.get_text()
    
    return full_text

studies = [
    "AI - Proof of concept (Study 1).pdf",
    "AI - Proof of concept (Study 2).pdf",
    "AI - Proof of concept (Study 3).pdf",
    "AI - Proof of concept (Study 4).pdf",
    "AI - Proof of concept (Study 5).pdf",
    "AI - Proof of concept (Study 6).pdf",
    "AI - Proof of concept (Study 7).pdf",
    "AI - Proof of concept (Study 8).pdf",
    "AI - Proof of concept (Study 9).pdf",
    "AI - Proof of concept (Study 10).pdf"
           ]

contexts = []

for studie in studies :
    context = pdf_to_str(studie)
    contexts.append(context)

# Liste des questions à poser
questions = [
    "Was it an in-vivo or an in-vitro experiment? if it was an in-vivo experiment, I want the last words of your answer to be : 'IN-VIVO - [Name_of_the_animal]' (replace [Name_of_the_animal] with the name of the animal). If it was an in-vitro experiment, I want the last words of your answer to be : 'IN-VITRO - [Name_of_the_strain]' (replace [Name_of_the_strain] with the name of the strain).",
    "If it was an oral exposure experiment, what type of exposure was it? If it was not an oral exposure experiment, the final word of your answer should be 'ORAL EXPOSURE : null' If it was an oral exposure experiment, the final word of your answer should be 'ORAL EXPOSURE : [Type_of_exposure]'",
    "If it was a dermal exposure experiment, what type of exposure was it? If it was not a dermal exposure experiment, the final word of your answer should be 'DERMAL EXPOSURE : null'. If it was a dermal exposure experiment, the final word of your answer should be 'DERMAL EXPOSURE : [Type_of_exposure]'",
    "If it was an inhalation exposure experiment, what type of exposure was it? If it was not an inhalation exposure experiment, the final word of your answer should be 'INHALATION EXPOSURE : null' If it was an inhalation exposure experiment, the final word of your answer should be 'INHALATION EXPOSURE : [Type_of_exposure]''.",
    "If the experiment was about sensitization, what was the output? If it was not a sensitization experiment, the final word of your answer should be 'SENSITIZATION : NA'. If it was a sensitization experiment, write the outcome in the following format : 'SENSITIZATION : Yes' if the results is positive (the compound causes sensitization), and 'SENSITIZATION : No' if the result is negative (the compound does not cause sensitization).",
    "If the experiment was about genotoxicity, what was the output? If it was not a genotoxicity experiment, the final word of your answer should be 'GENOTOXICITY : NA'. If it was a genotoxicity experiment, write the outcome in the following format : 'GENOTOXICITY : Yes' if the results is positive (the compound is genotoxic), and 'GENOTOXICITY : No' if the result is negative (the compound is not genotoxic).",
    "Was it a repeated dose experiment? If yes, the final word of your answer should be 'REPEATED DOSE : Yes'. If no, the final word of your answer should be 'REPEATED DOSE : No'.",
    "Was there any information about the purity of the compound tested? If yes, the final word of your answer should be 'PURITY : [purity]%''. If no, the final word of your answer should be 'PURITY : No info'",
    "What was the dilution of the compound that was used in this study? The last word of your anser should be 'DILUTION : [value_of_dilution]%''.",
    "Was the solvent used for dilution mentioned?  If yes, the final word of your answer should be 'SOLVENT : [solvent_name]''. If no, the final word of your answer should be 'SOLVENT : No info'.",
    "Was the number of animals used mentioned in the study? If yes, the final word of your answer should be 'NUMBER ANIMALS : [value]''. If no, the final word of your answer should be 'NUMBER ANIMALS : No info'."
]


# === Base URL for Ollama chat
url = "http://localhost:11434/api/chat"

# === Iterate over each context
for context_index, context in enumerate(contexts, 1):
    folder_name = f"Context_{context_index}"
    output_dir = os.path.join("conversations", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing {folder_name}")

    for i, question in enumerate(tqdm(questions, desc=folder_name, leave=False)):
        # Step 1: Send context
        context_data = {
            "model": "deepseek-r1:8b",
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
                    "model": "deepseek-r1:1.5b",
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