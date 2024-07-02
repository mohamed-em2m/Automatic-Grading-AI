## Team membmers:
- Mohamed Emam

- Abdelmonem Mansour

- Mariem Ahmed

- Mohamed Bahgat

- Ehab AbdelAzim

- Youssef sherif

## Model Description
![em5-english](https://github.com/mohamed-em2m/auto-grading/em5-english.png)

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
### Pipline
```python
from transformers import Pipeline
import torch.nn.functional as F


class MyPipeline:

    def __init__(self,model,tokenizer):
        self.model=model
        self.tokenizer=tokenizer

    def chat_Format(self,context, quetion, answer):
                        return "Instruction:/n check answer is true or false of next quetion using context below:\n" + "#context: " + context + f".\n#quetion: " + quetion + f".\n#student answer: " + answer + ".\n#response:"
                  

    def __call__(self, context, quetion, answer,generate=1,max_new_tokens=4, num_beams=2, do_sample=False,num_return_sequences=1):
                inp=self.chat_Format(context, quetion, answer)
                w = self.tokenizer(inp, add_special_tokens=True,
                                      pad_to_max_length=True,
                                      return_attention_mask=True,
                                      return_tensors='pt')
                response=""
                if(generate):
                    outputs = self.tokenizer.batch_decode(self.model.generate(input_ids=w['input_ids'].cuda(), attention_mask=w['attention_mask'].cuda(), max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=do_sample, num_return_sequences=num_return_sequences), skip_special_tokens=True)
                    response = outputs

                s =self.model(input_ids=w['input_ids'].cuda(), attention_mask=w['attention_mask'].cuda())['logits'][0][-1]
                s = F.softmax(s, dim=-1)
                yes_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("True")[0])
                no_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("False")[0])
                
                for i in  ["Yes", "yes", "True", "true","صحيح"]:
                  for word in self.tokenizer.tokenize(i): 
                    s[yes_token_id] += s[self.tokenizer.convert_tokens_to_ids(word)]
                for i in ["No", "no", "False", "false","خطأ"]:
                  for word in self.tokenizer.tokenize(i): 

                    s[no_token_id] += s[self.tokenizer.convert_tokens_to_ids(word)]
                true = (s[yes_token_id] / (s[no_token_id] + s[yes_token_id])).item()
                return {"response": response, "true": true}
context="""Large language models, such as GPT-4, are trained on vast amounts of text data to understand and generate human-like text. The deployment of these models involves several steps:

    Model Selection: Choosing a pre-trained model that fits the application's needs.
    Infrastructure Setup: Setting up the necessary hardware and software infrastructure to run the model efficiently, including cloud services, GPUs, and necessary libraries.
    Integration: Integrating the model into an application, which can involve setting up APIs or embedding the model directly into the software.
    Optimization: Fine-tuning the model for specific tasks or domains and optimizing it for performance and cost-efficiency.
    Monitoring and Maintenance: Ensuring the model performs well over time, monitoring for biases, and updating the model as needed.""" 
quetion="What are the key considerations when choosing a cloud service provider for deploying a large language model like GPT-4?"
answer="""When choosing a cloud service provider for deploying a large language model like GPT-4, the key considerations include:
    Compute Power: Ensure the provider offers high-performance GPUs or TPUs capable of handling the computational requirements of the model.
    Scalability: The ability to scale resources up or down based on the application's demand to handle varying workloads efficiently.
    Cost: Analyze the pricing models to understand the costs associated with compute time, storage, data transfer, and any other services.
    Integration and Support: Availability of tools and libraries that support easy integration of the model into your applications, along with robust technical support and documentation.
    Security and Compliance: Ensure the provider adheres to industry standards for security and compliance, protecting sensitive data and maintaining privacy.
    Latency and Availability: Consider the geographical distribution of data centers to ensure low latency and high availability for your end-users.

By evaluating these factors, you can select a cloud service provider that aligns with your deployment needs, ensuring efficient and cost-effective operation of your large language model."""
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM,AutoTokenizer

config = PeftConfig.from_pretrained("mohamedemam/Em2-bloomz-7b")
base_model = AutoModelForCausalLM.from_pretrained("mohamedemam/Em2-bloomz-7b")
model = PeftModel.from_pretrained(base_model, "mohamedemam/Em2-bloomz-7b")
tokenizer = AutoTokenizer.from_pretrained("mohamedemam/Em2-bloomz-7b", trust_remote_code=True)
pipe=MyPipeline(model,tokenizer)
print(pipe(context,quetion,answer,generate=True,max_new_tokens=4, num_beams=2, do_sample=False,num_return_sequences=1))
```
- **output:**{'response': ["Instruction:/n check answer is true or false of next quetion using context below:\n#context: Large language models, such as GPT-4, are trained on vast amounts of text data to understand and generate human-like text. The deployment of these models involves several steps:\n\n    Model Selection: Choosing a pre-trained model that fits the application's needs.\n    Infrastructure Setup: Setting up the necessary hardware and software infrastructure to run the model efficiently, including cloud services, GPUs, and necessary libraries.\n    Integration: Integrating the model into an application, which can involve setting up APIs or embedding the model directly into the software.\n    Optimization: Fine-tuning the model for specific tasks or domains and optimizing it for performance and cost-efficiency.\n    Monitoring and Maintenance: Ensuring the model performs well over time, monitoring for biases, and updating the model as needed..\n#quetion: What are the key considerations when choosing a cloud service provider for deploying a large language model like GPT-4?.\n#student answer: When choosing a cloud service provider for deploying a large language model like GPT-4, the key considerations include:\n    Compute Power: Ensure the provider offers high-performance GPUs or TPUs capable of handling the computational requirements of the model.\n    Scalability: The ability to scale resources up or down based on the application's demand to handle varying workloads efficiently.\n    Cost: Analyze the pricing models to understand the costs associated with compute time, storage, data transfer, and any other services.\n    Integration and Support: Availability of tools and libraries that support easy integration of the model into your applications, along with robust technical support and documentation.\n    Security and Compliance: Ensure the provider adheres to industry standards for security and compliance, protecting sensitive data and maintaining privacy.\n    Latency and Availability: Consider the geographical distribution of data centers to ensure low latency and high availability for your end-users.\n\nBy evaluating these factors, you can select a cloud service provider that aligns with your deployment needs, ensuring efficient and cost-effective operation of your large language model..\n#response:  true the answer is"], 'true': 0.943033754825592}

### Chat Format Function
This function formats the input context, question, and answer into a specific structure for the model to process.

```python
def chat_Format(self, context, question, answer):
    return "Instruction:/n check answer is true or false of next question using context below:\n" + "#context: " + context + f".\n#question: " + question + f".\n#student answer: " + answer + ".\n#response:"
```
