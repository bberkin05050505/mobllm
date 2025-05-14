from openai import OpenAI
from textgrad.engine.local_model_openai_api import ChatExternalClient
import textgrad as tg

from models.koboldcpp_openai_model import KoboldOpenAIModel
import torch


def textGrad_test():
    # start a server with lm-studio and point it to the right address; here we use the default address. 
    client = OpenAI(base_url="http://localhost:5001/v1", api_key="local_llm")
    engine = ChatExternalClient(client=client, model_string='koboldcpp')

    # Set this engine as the backward engine for TextGrad
    tg.set_backward_engine(engine, override=True)

    # Create a variable with your prompt
    question = tg.Variable("5+3+3+3+(-3)=? Here's the solution: 5+3+3+3=101..", 
                        requires_grad=True, 
                        role_description="confirming answer to a math question")

    loss_system_prompt = tg.Variable("""You will evaluate a solution to a math question. 
    Do not attempt to solve it yourself, do not give a solution, only identify errors. Be super concise.""",
                                    requires_grad=False,
                                    role_description="system prompt")
                                
    loss_fn = tg.TextLoss(loss_system_prompt)
    optimizer = tg.TGD([question])

    # This computes the loss, i.e. the textual gradient, the LLM critique
    loss = loss_fn(question)
    print("Loss: ", loss.value)

    # This creates a new prompt including what to do, what to correct etc. based on the loss
    loss.backward()

    # This creates an improved version of the original variable according to the suggestions received in loss.backward()
    optimizer.step()
    print("Optimized question: ", question.value)

def textGrad_easy_test():
    client = OpenAI(base_url="http://localhost:5001/v1", api_key="local_llm")
    engine = ChatExternalClient(client=client, model_string='koboldcpp')
    print(engine.generate(max_tokens=50, content="What is the meaning of life???>>>"))
    # turns out stop=None was causing issues and somehow resulting in None responses from the server

def test_local_model_openAI_api():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    trial = KoboldOpenAIModel("koboldcpp", device, dtype)
    response = trial.generate("What is the meaning of life?")
    print(response)

if __name__ == "__main__":
    textGrad_test()



     


