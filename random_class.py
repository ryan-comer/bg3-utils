import random as rand
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

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
        self.model_name = "TheBloke/Llama-2-7B-Chat-GPTQ"
        self.initialize_model()

    def initialize_model(self):
        print("Initializing model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto')
        self.model.generation_config = GenerationConfig(eos_token_id=self.tokenizer.eos_token_id, max_new_tokens=1000, pad_token_id=self.tokenizer.eos_token_id, num_beams=5, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.2, length_penalty=1.0, no_repeat_ngram_size=3, num_return_sequences=1)

    def add_class(self, name):
        self.classes.append(Class(name))

    def get_random_class(self):
        return rand.choice(self.classes)

    # Generate text from a prompt
    def generate_text(self, prompt):
        # Tokenize the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        # Generate the text
        outputs = self.model.generate(inputs)
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
    generator.add_class("Barbarian")
    generator.add_class("Bard")
    generator.add_class("Cleric")
    generator.add_class("Druid")
    generator.add_class("Fighter")
    generator.add_class("Monk")
    generator.add_class("Paladin")
    generator.add_class("Ranger")
    generator.add_class("Rogue")
    generator.add_class("Sorcerer")
    generator.add_class("Warlock")
    generator.add_class("Wizard")

    # Generate 3 random classes and store in an array with no duplicates
    classes = []
    while len(classes) < 3:
        c = generator.get_random_class()
        if c not in classes:
            classes.append(c)    

    # Generate the details for the classes
    details = generator.get_class_group_details(classes)

    # Print the details
    print ('\n\n\n----------------------------------------------------------------\n\n\n')
    print("Classes: " + ", ".join([c.name for c in details.classes]) + "\n")
    print("Name: " + details.name + "\n")
    print("Description: " + details.description + "\n")
    print("Playstyle: " + details.playstyle)
    print ('\n\n\n----------------------------------------------------------------\n\n\n')