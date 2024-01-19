from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
from random_class import ClassGenerator
import uuid

app = Flask(__name__)
cors = CORS(app)

class_names = [
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

@app.route('/')
def test():
    return 'Hello World!'

@app.route('/images/<image_id>')
@cross_origin()
def get_image(image_id):
    return send_file(f"images/{image_id}.png", mimetype='image/png')

@app.route('/api/v1/generate_class', methods=['POST'])
@cross_origin()
def generate_class():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        # Parse the inputs
        json = request.get_json()
        num_characters = json['num_characters']

        # Check if the number of characters is an integer
        if not isinstance(num_characters, int):
            return 'Bad Request', 400

        # Initialize the generator with classes
        generator = ClassGenerator()
        for name in class_names:
            generator.add_class(name)

        # Initialize the generator for text generation
        generator.initialize_text_generation_model()

        # Generate the classes
        response = []
        portrait_prompts = []
        for i in range(num_characters):
            print(f"Generating text for character {i}...")
            # Generate 3 random classes and store in an array with no duplicates
            classes = generator.get_random_classes(3)

            # Generate the name
            name = generator.generate_name(classes)
            description = generator.generate_description(classes)
            portrait_prompt = generator.generate_portrait_prompt(classes)
            
            # Add to the response
            response.append({
                'name': name,
                'description': description,
                'classes': [
                    {
                        "name": c.name,
                        "image": '/class_icons/' + c.name.lower() + '_icon.png'
                    }
                    for c in classes
                ]
            })

            # Add to the portrait prompts
            portrait_prompts.append(portrait_prompt)

        generator.cleanup_models()
        generator.initialize_image_generation_model()

        # Generate the portraits
        for i in range(num_characters):
            print(f"Generating portrait for character {i}...")
            # Generate the portrait
            portrait = generator.generate_image(portrait_prompts[i])
            
            # Generate a random GUID for the image
            image_id = str(uuid.uuid4())
            portrait.save(f"images/{image_id}.png")
            response[i]['image_id'] = image_id

        # Return the response
        generator.cleanup_models()
        print(response)
        return response, 200

    else:
        return 'Unsupported Media Type', 415

if __name__ == '__main__':
    app.run(debug=True)