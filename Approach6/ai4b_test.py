import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

# Load the model and tokenizer
model_dir = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
ip = IndicProcessor(inference=True)

# Define a simple translation function
def test_translation(text, src_lang="eng_Latn", tgt_lang="hin_Deva"):
    # Preprocess the text
    batch = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=5)
    
    # Decode and post-process the translation
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    translated_text = ip.postprocess_batch(decoded, lang=tgt_lang)
    return translated_text[0]

# Test with a simple sentence
test_sentence = "The ThawEasy Lite machine does not have a temperature setting."
translated_output = test_translation(test_sentence)
print(f"Translated Text (Hindi): {translated_output}")
