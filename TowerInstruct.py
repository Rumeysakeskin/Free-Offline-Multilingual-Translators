# # Install transformers from source - only needed for versions <= v4.34
# # pip install git+https://github.com/huggingface/transformers.git
# # pip install accelerate

# import torch
# from transformers import pipeline

# pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")
# # We use the tokenizer’s chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
# messages = [
#     {"role": "user", "content": "Translate the following text from English into Turkish.\nEnglish: A group of researchers has launched a new model for translation-related tasks..\nTurkish:"},
# ]
# prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
# print(outputs[0]["generated_text"])
# # <|im_start|>user
# # Translate the following text from Portuguese into English.
# # Portuguese: Um grupo de investigadores lançou um novo modelo para tarefas relacionadas com tradução.
# # English:<|im_end|>
# # <|im_start|>assistant
# # A group of researchers has launched a new model for translation-related tasks.


# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline

def translate_TowerInstruct(input_text, source_language, target_language):
    pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-v0.1", torch_dtype=torch.bfloat16, device_map="auto")

    # Form the message with the user input
    messages = [
        {
            "role": "user",
            "content": f"Translate the following text from {source_language} into {target_language}.\n{source_language}: {input_text}\n{target_language}:"
        }
    ]

    # Generate the prompt using the tokenizer’s chat template
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate the output
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False)

    # Return the generated translation
    # return outputs[0]["generated_text"]
      
    generated_text = outputs[0]["generated_text"]

    assistant_start = generated_text.find("<|im_start|>assistant") + len("<|im_start|>assistant")
    
    return generated_text[assistant_start:].strip()

# Dil kodları ve isimleri için dict
language_dict = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "zh": "Chinese",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "ko": "Korean",
    "it": "Italian",
    "es": "Spanish",
    "tr": "Turkish"
}

# Dil koduna göre dil ismini döndüren fonksiyon
def get_language_name(lang_code):
    return language_dict.get(lang_code, "Unknown Language")

# Örnek kullanım
source_lang_code = "tr"
target_lang_code = "en"
input_text = "Bir grup araştırmacı, çeviri ile ilgili görevler için yeni bir model başlattı."

source_lang = get_language_name(source_lang_code)
target_lang = get_language_name(target_lang_code)

translated_text = translate_TowerInstruct(input_text, source_lang, target_lang)
print(translated_text)

