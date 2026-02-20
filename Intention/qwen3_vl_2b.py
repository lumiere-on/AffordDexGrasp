# pip install git+https://github.com/huggingface/transformers
# pip install transformers==4.57.0 # currently, V4.57.0 is not released

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-2B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

system_prompt = """
Your task is to guide the robot to execute user commands. Firstly, accept human commands, then analyze the images and output information in a specified format to guide the robot to grasp and manipulate objects. Please analyze step by step in the following order:
1. What category does the object in the picture belong to
2. What is the intention of grabbing (use/hold/lift/hand over)
3. What part of this object should be grasped (e.g. handle or body)
4. In which direction should this part be grasped (clockwise from red to blue indicates grasping from above, grasping from right, grasping from front, grasping from left). 
5. Final guidance: the [part] from [direction] by contacting [surface]. 

And output: 
Object category: {} 
Capture intention: {} 
Grasping object part: {} 
Grasping direction: {} 
Final guidance: {}
"""

user_instruction="" # use/hold/lift/hand over
image_url = ""


messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_url,
            },
            {
                "type": "text", 
                "text": f"User Command: {user_instruction}\n\nPlease analyze the image and provide the output in the specified format."                },
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
