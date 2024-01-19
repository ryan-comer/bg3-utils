import random as rand
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from diffusers import DiffusionPipeline

# A single class
class Class:
    def __init__(self, name):
        self.name = name

# Details for a group of classes
class ClassGroupDetails:
    def __init__(self, name, description, playstyle, classes):
        self.name = name
        self.description = description
        self.classes = classes
        self.playstyle = playstyle

# Generator for classes
class ClassGenerator:
    def __init__(self):
        self.classes = []

        # Used for text generation
        self.model_name_text_generation = "TheBloke/Llama-2-7B-Chat-GPTQ"
        self.model_text_generation = None
        self.tokenizer = None

        # Used for image generation
        self.model_name_image_generation = 'stabilityai/stable-diffusion-xl-base-1.0'
        self.model_image_generation = None

    # Initialize the model for text generation
    def initialize_text_generation_model(self):
        print("Initializing text generation model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_text_generation)
        self.model_text_generation = AutoModelForCausalLM.from_pretrained(self.model_name_text_generation, device_map='auto')
        self.model_text_generation.generation_config = GenerationConfig(eos_token_id=self.tokenizer.eos_token_id, max_new_tokens=1000, pad_token_id=self.tokenizer.eos_token_id, num_beams=5, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.2, length_penalty=1.0, no_repeat_ngram_size=3, num_return_sequences=1)

    # Initialize the model for image generation
    def initialize_image_generation_model(self):
        print("Initializing image generation model...")
        self.model_image_generation = DiffusionPipeline.from_pretrained(self.model_name_image_generation, torch_dtype=torch.bfloat16, use_safetensors=True)
        self.model_image_generation.to('cuda')
        self.model_image_generation.enable_xformers_memory_efficient_attention()


    # Clean up any active models
    def cleanup_models(self):
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.model_text_generation is not None:
            del self.model_text_generation
            self.model_text_generation = None

        if self.model_image_generation is not None:
            del self.model_image_generation
            self.model_image_generation = None
        
        torch.cuda.empty_cache()

    def add_class(self, name):
        self.classes.append(Class(name))

    def get_random_classes(self, num_classes):
        return rand.sample(self.classes, num_classes)

    def generate_image(self, prompt):
       image = self.model_image_generation(prompt, num_inference_steps=20).images[0]
       return image

    # Generate text from a prompt
    def generate_text(self, prompt):
        # Tokenize the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model_text_generation.device)
        # Generate the text
        outputs = self.model_text_generation.generate(inputs)
        # Decode the text
        text = self.tokenizer.batch_decode(outputs[:, inputs.shape[1]:])[0]
        text = text.replace("</s>", "")
        # Return the text
        return text

    # Generate the name for a group of classes
    def generate_name(self, classes):
        prompt = """
            I have a list of 3 classes from dungeons and dragons.
            I'm making a new class that is a combination of these 3.
            I want you to give me a name for this new class.
            Make the name make sense based on the 3 classes.
            Only give me the name.
            Don't say anything that isn't the name.
            An example output for the inputs "Wizard, Barbarian, Rogue" would be "Holy Ravager" because this incorporates themes from each of the 3 classes.
            An example output for the inputs "Barbarian, Druid, Wizard" would be "Primal Arcanewarden" because this incorporates themes from each of the 3 classes.
            An example output for the inputs "Monk, Rogue, Sorcerer" would be "Ethereal Flowstriker" because this incorporates themes from each of the 3 classes.
            Don't copy these examples exactly, but use them as a guide for capturing the theme of the original 3 classes.
            Surround the actual name with asterisks.
            You are only giving the name, don't say any other text to clarify or explain.
            Here are the 3 classes: """ + ", ".join([c.name for c in classes]) + ".\n"
        prompt_template = """
            [INST] <<SYS>>
            You are an assistant that is used to generate metadata for dungeons and dragon classes. Be crisp and clear in your responses. Only reply with what your are asked for, nothing else.
            <</SYS>>
            {}[/INST]
        """

        # Generate the text
        text = self.generate_text(prompt_template.format(prompt))

        # The name is surrounded by asterisks, remove them
        return text.replace("*", "")

    # Generate a description for a group of classes
    def generate_description(self, classes):
        prompt = """
            I have a list of 3 classes from dungeons and dragons.
            I'm making a new class that is a combination of these 3.
            I want you to give me a description for this new class.
            Make the description make sense based on the 3 classes.
            Only give me the description.
            Don't say anything that isn't the description.
            You are only giving the description, don't say any other text to clarify or explain.
            Just say the description.
            Here are the 3 classes: """ + ", ".join([c.name for c in classes]) + ".\n"
        prompt_template = """
            [INST] <<SYS>>
            You are an assistant that is used to generate metadata for dungeons and dragon classes. Be crisp and clear in your responses. Only reply with what your are asked for, nothing else.
            <</SYS>>
            {}[/INST]
            The description for the new class is: 
        """

        # Generate the text
        return self.generate_text(prompt_template.format(prompt))
    
    # Generate a playstyle for a group of classes
    def generate_playstyle(self, classes):
        prompt = """
            I have a list of 3 classes from dungeons and dragons.
            I'm making a new class that is a combination of these 3.
            I want you to give me a playstyle for this new class.
            The playstayle is how the player should play the new class while playing dungeons and dragons.
            Give the player tips on what they should do while playing the new class. Say what the strengths and weaknesses of the new class are.
            Make the playstyle make sense based on the 3 classes.
            Only give me the playstyle.
            Don't say anything that isn't the playstyle.
            You are only giving the playstyle, don't say any other text to clarify or explain.
            Just say the playstyle.
            Here are the 3 classes: """ + ", ".join([c.name for c in classes]) + ".\n"
        prompt_template = """
            [INST] <<SYS>>
            You are an assistant that is used to generate metadata for dungeons and dragon classes. Be crisp and clear in your responses. Only reply with what your are asked for, nothing else.
            <</SYS>>
            {}[/INST]
            The playstyle for the new class is: 
        """

        # Generate the text
        return self.generate_text(prompt_template.format(prompt))

    # Generate a list of descriptors for a class
    def generate_class_descriptors(self, character_class):
        prompt = """
            I have a class from dungeons and dragons.
            I want you to give me a list of descriptors for this class.
            The descriptors should describe the class.
            The descriptors should be separated by commas.
            Here is an example for the input "Wizard": "magical energy, fire, ice, staff, robes, old, beard".
            Here is an example for the input "Barbarian": "muscles, strong, rage, battleaxe, fighting".
            Here is an example for the input "Rogue": "sneaky, stealthy, daggers, shadows, blood".
            Your output should only be a comma separated list of descriptors.
            Your output should only be a comma separated list of descriptors.
            Your output should only be a comma separated list of descriptors.
            Only give me the descriptors.
            Don't say anything that isn't the descriptors.
            Here is the class: """ + character_class.name + ".\n"

        prompt_template = """
            [INST] <<SYS>>
            You are an assistant that is used to generate metadata for dungeons and dragon classes. Be crisp and clear in your responses. Only reply with what your are asked for, nothing else.
            <</SYS>>
            {}[/INST]
            The prompt for the new class is: 
        """

        return self.generate_text(prompt_template.format(prompt))

    # Generate a portrait prompt for a group of classes
    def generate_portrait_prompt(self, classes):
        descriptors = []
        for character_class in classes:
            descriptors += self.generate_class_descriptors(character_class).split(", ")[1:]

        descriptors = [d.strip() for d in descriptors if d.strip() != ""]

        # Get a random subset of the descriptors
        rand.shuffle(descriptors)
        if len(descriptors) > 6:
            descriptors = descriptors[:4]
        else:
            descriptors = descriptors[:len(descriptors)]

        prompt = f"A character portrait for dungeons and dragons, single character, color, {','.join(descriptors)}, high detail, high quality"
        return prompt

    # Generate the details for a group of classes
    def get_class_group_details(self, classes):
        # Generate the name
        print("Generating name...")
        name = self.generate_name(classes)
        # Generate the description
        print("Generating description...")
        description = self.generate_description(classes)
        # Generate the playstyle
        print("Generating playstyle...")
        playstyle = self.generate_playstyle(classes)

        # Create the class group details
        return ClassGroupDetails(name, description, playstyle, classes)

if __name__ == "__main__":
    # Create the generator
    generator = ClassGenerator()

    # Add all the classes
    names = [
        "Barbarian",
        "Bard",
        "Cleric",
        "Druid",
        "Fighter",
        "Monk",
        "Paladin",
        "Ranger",
        "Rogue",
        "Sorcerer",
        "Warlock",
        "Wizard"
    ]
    for name in names:
        generator.add_class(name)

    num_characters = 4
    for i in range(num_characters):
        # Generate 3 random classes and store in an array with no duplicates
        classes = generator.get_random_classes(3)

        # Generate the name
        name = generator.generate_name(classes)
        description = generator.generate_description(classes)

        # Print the details
        print ('\n----------------------------------------------------------------\n')
        print("Classes: " + ", ".join([c.name for c in classes]) + "\n")
        print("Name: " + name + "\n")
        print("Description: " + description + "\n")
        print ('\n----------------------------------------------------------------\n')