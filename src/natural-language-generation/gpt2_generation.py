from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
import logging
import sys

# 0) SETUP
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 1) LOAD FINE-TUNED MODEL AND TOKENIZER
checkpoint = sys.argv[1]  # path to a fine-tuned model
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=checkpoint)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=checkpoint,
                                             config=config)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint)

# 2) CREATE A PROMPT, TOKENIZE IT AND CREATE TENSORS
while True:
    prompt = input('\n\nInsert prompt\n')
    max_length = int(input('\nInsert max generation length\n'))

    tokenized_prompt = tokenizer(prompt, return_tensors='pt')

    # 3) RUN CONDITIONAL GENERATION
    print(f"Run conditional generation with prompt: <{prompt}>")
    output_sentence = model.generate(input_ids=tokenized_prompt['input_ids'],
                                     max_length=max_length,
                                     top_p=0.9,
                                     top_k=100,
                                     do_sample=True)

    output_sentence.squeeze_()  # remove batch dimension
    generated_text = tokenizer.decode(output_sentence)
    # TODO: add postprocessing to clean the generated text (e.g, cut the text at stop words such as periods)
    print(f"Generated recipe:\n {generated_text}")
