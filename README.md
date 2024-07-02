## Team membmers:
- Mohamed Emam

- Abdelmonem Mansour

- Mariem Ahmed

- Mohamed Bahgat

- Ehab AbdelAzim

- Youssef sherif

## Model Description

We are thrilled to introduce our graduation project, the EM2 model, designed for automated essay grading in both Arabic and English. 📝✨

To develop this model, we first created a custom dataset for training. We adapted the QuAC and OpenOrca datasets to make them suitable for our automated essay grading application.

Our model utilizes the following impressive models:

- Mistral: 96%
- LLaMA: 93%
- FLAN-T5: 93%
- BLOOMZ (Arabic): 86%
- MT0 (Arabic): 84%

You can try our models for auto-grading on Hugging Face! 🌐

We then deployed these models for practical use. We are proud of our team's hard work and the potential impact of the EM2 model in the field of education. 🌟

#MachineLearning #AI #Education #EssayGrading #GraduationProject


## How It Works

The model takes three inputs: the context (or perfect answer), a question on the context, and the student answer. The model then outputs the result.

- Repository: [https://github.com/mohamed-em2m/auto-grading](https://github.com/mohamed-em2m/auto-grading)

### Direct Use

Auto grading for essay questions.

### Downstream Use [optional]

Text generation.

## Training Data

- Dataset: `mohamedemam/Essay-quetions-auto-grading-arabic`

## Training Procedure

Using TRL.

### Pipeline

```python
from transformers import Pipeline
import torch.nn.functional as F

class MyPipeline:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def chat_Format(self, context, question, answer):
        return ("Instruction:/n check answer is true or false of next question using context below:\n" 
                + "#context: " + context + f".\n#question: " + question 
                + f".\n#student answer: " + answer + ".\n#response:")

    def __call__(self, context, question, answer, generate=1, max_new_tokens=4, num_beams=2, do_sample=False, num_return_sequences=1):
        inp = self.chat_Format(context, question, answer)
        w = self.tokenizer(inp, add_special_tokens=True,
                           pad_to_max_length=True,
                           return_attention_mask=True,
                           return_tensors='pt')
        response = ""
        if generate:
            outputs = self.tokenizer.batch_decode(
                self.model.generate(input_ids=w['input_ids'].cuda(), attention_mask=w['attention_mask'].cuda(),
                                    max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=do_sample,
                                    num_return_sequences=num_return_sequences), skip_special_tokens=True)
            response = outputs

        s = self.model(input_ids=w['input_ids'].cuda(), attention_mask=w['attention_mask'].cuda())['logits'][0][-1]
        s = F.softmax(s, dim=-1)
        yes_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("True")[0])
        no_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("False")[0])

        for i in ["Yes", "yes", "True", "true", "صحيح"]:
            for word in self.tokenizer.tokenize(i):
                s[yes_token_id] += s[self.tokenizer.convert_tokens_to_ids(word)]
        for i in ["No", "no", "False", "false", "خطأ"]:
            for word in self.tokenizer.tokenize(i):
                s[no_token_id] += s[self.tokenizer.convert_tokens_to_ids(word)]
        true = (s[yes_token_id] / (s[no_token_id] + s[yes_token_id])).item()
        return {"response": response, "true": true}
