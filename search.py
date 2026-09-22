import sys
sys.path.insert(0, '.')
from build_index import documents

for doc in documents:
    if "99.54" in doc["text"]:
        idx = doc["text"].find("99.54")
        print(f"Found in: {doc['source']}")
        print(doc["text"][max(0, idx-300):idx+300])
        print("---")
    if "97.85" in doc["text"]:
        idx = doc["text"].find("97.85")
        print(f"Found in: {doc['source']}")
        print(doc["text"][max(0, idx-300):idx+300])
        print("---")


# answer for the test model:
# Both `max_new_tokens` (=150) and `max_length`(=131072) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
# <|begin_of_text|>Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# What features are used to detect audio deepfakes?

# ### Response:
# MFCC, LFCC, Chroma, Mel-Frequency Cepstral Coefficients (MFCC), and Chroma Features are used to detect audio deepfakes.<|eot_id|>
# Colab paid products
# -
# Cancel contracts here
