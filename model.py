import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"

model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_text(prompt, max_length = 1, temperature=0.7, top_k=50, repetition_penalty=1.2):
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        num_return_sequences=1,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    prompt = input("Enter the initial prompt: ")
    max_length = int(input("Enter max length of generated text (e.g., 200): "))
    
    temperature = 0.7
    top_k = 50
    repetition_penalty = 1.2
    
    generated_text = generate_text(prompt, max_length, temperature, top_k, repetition_penalty)
    
    print("\nGenerated Text:\n")
    print(generated_text)
    
    save_option = input("\nDo you want to save the generated text? (yes/no): ").strip().lower()
    if save_option == 'yes':
        file_name = input("Enter the file name (without extension): ").strip()
        with open(f"{file_name}.txt", "w") as file:
            file.write(generated_text)
        print(f"Text saved as {file_name}.txt")

if __name__ == "__main__":
    main()
