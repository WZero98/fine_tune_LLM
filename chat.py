# |--File name:   --|chat
# |--File path:   --|dataset
# |--Create Date: --|2025/9/2
# |--Programmer:  --|PY.Wang
# |--Description: --|Description
# |--version:     --|
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading


def text_generate(input: str, model, tokenizer) -> str:
    # prepare the model input
    prompt = input
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 定义 streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 生成放到后台线程
    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(**model_inputs, streamer=streamer, max_new_tokens=32768)
    )
    thread.start()

    # 主线程里迭代获取流式结果
    output = ''
    for new_text in streamer:
        output += new_text
        print(new_text, end="", flush=True)

    return output


def text_generate_no_stream(input: str, model, tokenizer) -> tuple[str, str]:
    # prepare the model input
    prompt = input
    messages = [
        {"role": "user", "content": f"{prompt}"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )

    # 生成的token id 的起始处包含了input token
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content) # no opening <think> tag
    print("content:", content)
    return (thinking_content, content)



if __name__ == "__main__":
    model_path = "saved_models/Qwen3-0.6B"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map=device
    )
    answer_1 = text_generate('你是谁', model, tokenizer)
    answer_2 = text_generate_no_stream('你是谁', model, tokenizer)
