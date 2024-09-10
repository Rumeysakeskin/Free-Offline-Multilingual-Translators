import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class OPUSTranslator:
    def __init__(self, device="cpu", quant4=False, quant4_config=None, quant8=False, max_length=512):
        self.device = torch.device(device)
        self.quant4 = quant4
        self.quant4_config = quant4_config
        self.quant8 = quant8
        self.max_length = max_length
        self.model_cache = {}
        self.alternative_models = {
            "en-pl": 'gsarti/opus-mt-tc-en-pl',
            "en-ja": 'gsarti/opus-mt-tc-base-en-ja'
        }

    def load_model(self, model_name, model_key):
        """ Load model and tokenizer for the given model name with optional quantization """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.quant4:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=self.device, quantization_config=self.quant4_config, load_in_4bit=True)
        elif self.quant8:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=self.device, load_in_8bit=True)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        self.model_cache[model_key] = (model, tokenizer)
        return model, tokenizer

    def get_helsinki_nlp_model(self, source_lang, target_lang):
        """ Get the OPUS translation model from Helsinki-NLP for a given language pair """
        if source_lang == 'pt-BR':
            source_lang = 'bzs'
        if source_lang == 'uk-UA':
            source_lang = 'uk'
        model_key = f'{source_lang}-{target_lang}'

        if model_key in self.model_cache:
            return self.model_cache[model_key]

        # Try standard OPUS model first
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        try:
            return self.load_model(model_name, model_key)
        except OSError:
            # Try tc-big naming convention
            try:
                model_name = f'Helsinki-NLP/opus-mt-tc-big-{source_lang}-{target_lang}'
                return self.load_model(model_name, model_key)
            except OSError:
                # Try alternative models (like gsarti/opus-mt-tc-en-pl)
                try:
                    model_name = self.alternative_models[model_key]
                    return self.load_model(model_name, model_key)
                except Exception:
                    return None, None

    def translate(self, texts, source_lang, target_lang):
        """ Translate texts from source_lang to target_lang with fallback to intermediate translation """
        with torch.no_grad():
            model, tokenizer = self.get_helsinki_nlp_model(source_lang, target_lang)
            if model is None or tokenizer is None:
                # Try translating through an intermediate language (English)
                model_i, tokenizer_i = self.get_helsinki_nlp_model(source_lang, 'en')
                model_t, tokenizer_t = self.get_helsinki_nlp_model('en', target_lang)

                if model_i is None or model_t is None:
                    print(f"No translation possible from {source_lang} to {target_lang}")
                    return None

                # Translate to intermediate language first (e.g., source_lang -> en)
                inputs = tokenizer_i(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                intermediate_outputs = model_i.generate(inputs.input_ids, max_length=self.max_length)
                intermediate_texts = [tokenizer_i.decode(output, skip_special_tokens=True) for output in intermediate_outputs]

                # Translate from intermediate language to target_lang (e.g., en -> target_lang)
                inputs = tokenizer_t(intermediate_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                translated_outputs = model_t.generate(inputs.input_ids, max_length=self.max_length)
                translated_texts = [tokenizer_t.decode(output, skip_special_tokens=True) for output in translated_outputs]

                return translated_texts
            else:
                # Direct translation (source_lang -> target_lang)
                inputs = tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                outputs = model.generate(inputs.input_ids, max_length=self.max_length)
                translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

                return translated_texts


if __name__ == '__main__':
    # Instantiate the translator with optional quantization
    translator = OPUSTranslator(device="cuda" if torch.cuda.is_available() else "cpu", quant4=False, quant8=False, max_length=512)

    source_texts = ["sana yarın gelmen gerektiğini söylemiştim", "burada ne işin var"]
    translated_texts = translator.translate(source_texts, "tr", "en")
    
    # Print the translated texts
    if translated_texts:
        for i, text in enumerate(translated_texts):
            print(f"Sentence {i + 1}: {text}")
